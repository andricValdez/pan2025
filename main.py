import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import pandas as pd
import numpy as np
import time
import warnings
import nltk 
from itertools import chain
import re
from spacy.tokens import Doc
import spacy
from tqdm import tqdm 
from torch_geometric.utils.convert import from_networkx
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    GlobalAttention, Set2Set
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
from torch_geometric.nn import MLP
from functools import lru_cache
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc
import mlflow
from mlflow import MlflowClient
import matplotlib.pyplot as plt 
import seaborn as sns
import argparse
import json
import sys
import os
import ast  

import utils
import baseline

# Configs
os.environ["HF_HUB_OFFLINE"] = "1"

warnings.filterwarnings("ignore")
log_file_path = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s; - %(levelname)s; - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load spaCy tokenizer ---
nlp = spacy.load("en_core_web_sm", disable=[])


# ****************** POS_TAGS
POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN", "AUX", "CCONJ", "SCONJ", 
    "ADP", "DET", "NUM", "PART", "INTJ", "PUNCT", "SYM", "X", "ROOT_0"
]
POS2IDX = {pos: idx for idx, pos in enumerate(POS_TAGS)}
NUM_POS_TAGS = len(POS_TAGS)

# ****************** DEPENDENCY_TAGS
DEPENDENCY_TAGS = [
    'ROOT', 'nsubj', 'obj', 'iobj', 'obl', 'nmod', 'amod', 'advmod', 'aux',
    'cop', 'mark', 'cc', 'conj', 'det', 'case', 'compound', 'xcomp', 'ccomp',
    'acl', 'appos', 'nummod', 'punct', 'dep', 'parataxis'
]
DEP2IDX = {tag: idx for idx, tag in enumerate(DEPENDENCY_TAGS)}
NUM_DEP_TAGS = len(DEPENDENCY_TAGS)


def get_multilevel_lang_features(doc) -> list:
    """Extract multilevel features from a spaCy Doc or Span object (e.g., sentence)"""
    doc_tokens = [] 
    for token in doc:
        #print(f"'{token.text}'", end=' | ')
        #synonyms_token = get_synonyms(token.text)
        #synonyms_token_head = get_synonyms(token.head.text)

        token_info = {
            'token': token,
            'token_text': token.text,
            'token_lemma': token.lemma_,
            'token_pos': token.pos_,
            'token_dependency': token.dep_,
            'token_head': token.head,
            'token_head_text': token.head.text,
            'token_head_lemma': token.head.lemma_,
            'token_head_pos': token.head.pos_,
            #'token_synonyms': synonyms_token[:5],
            #'token_head_synonyms': synonyms_token_head[:5],
            'is_root_token': token.dep_ == 'ROOT'
        }
        doc_tokens.append(token_info)

    return doc_tokens


class ISG():
    def __init__(self, 
                graph_type,
                output_format='', 
                apply_prep=True, 
                parallel_exec=False, 
                language='en', 
                steps_preprocessing={}
            ):
        self.apply_prep = apply_prep
        self.parallel_exec = parallel_exec
        self.graph_type = graph_type
        self.output_format = output_format

    def _get_entities(self, text_doc: list) -> list:  
        nodes = [('ROOT_0', {'pos_tag': 'ROOT_0'})]
        for d in text_doc:
            token_text = d['token_text'].lower()
            token_pos = d['token_pos']
            key = f"{token_text}_{token_pos}"
            nodes.append((key, {'pos_tag': token_pos}))
        return nodes

    def _get_relations(self, text_doc: list) -> list:
        edges = []

        for d in text_doc:
            word = d['token_text'].lower()
            head_word = d['token_head_text'].lower()
            pos = d['token_pos']
            head_pos = d['token_head_pos']
            dep = d['token_dependency']
            distance = abs(d['token'].i - d['token'].head.i)
            norm_dist = 1.0 / distance if distance > 0 else 1.0

            edge_attr = {"gramm_relation": dep, "token_distance": norm_dist}
            if d['is_root_token']:
                edges.append(('ROOT_0', f"{word}_{pos}", edge_attr))
            else:
                edges.append((f"{head_word}_{head_pos}", f"{word}_{pos}", edge_attr))
        return edges

    def _build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'undirected':
            graph = nx.Graph()
        else:
            graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _get_frequency_weight(self, graph: nx.DiGraph):
        freq_dict = defaultdict(int)    
        for edge in graph.edges(data=True):
            freq_dict[edge[2]['gramm_relation']] += 1

        for edge in graph.edges(data=True):
            edge[2]['gramm_relation'] = f'{edge[2]["gramm_relation"]}_{freq_dict[edge[2]["gramm_relation"]]+graph.number_of_edges(edge[0], edge[1])}'

    def _build_ISG_graph(self, graphs: list) -> networkx:
        if self.graph_type == 'undirected':
            int_synt_graph = nx.Graph()
        else:
            int_synt_graph = nx.DiGraph()

        for graph in graphs:
            int_synt_graph = nx.compose(int_synt_graph, graph)
        return int_synt_graph

    def _build_ISG_graph_v2(self, graphs: list) -> nx.DiGraph:
        G = nx.DiGraph() if self.graph_type == 'directed' else nx.Graph()
        for g in graphs:
            G.add_nodes_from(g.nodes(data=True))
            G.add_edges_from(g.edges(data=True))
        return G

    def _transform_pipeline(self, docs: list, set: str) -> list:
        doc_graphs = []
        for doc in tqdm(docs, desc=f"Building ISG per doc for {set} set"):
            try:
                sentence_graphs = []
                #print("\n ************************************")
                #print(doc.text)
                for sent in doc.sents:
                    #print("\n", sent)
                    sent_info = get_multilevel_lang_features(sent)
                    #print("\n", sent_info)
                    nodes = self._get_entities(sent_info)
                    edges = self._get_relations(sent_info)
                    graph = self._build_graph(nodes, edges)
                    #print(graph.nodes())
                    #print(graph.edges(data=True))
                    sentence_graphs.append(graph)
                doc_graph = self._build_ISG_graph(sentence_graphs)
                #print(doc_graph.nodes())
                #print(doc_graph.edges())
                doc_graphs.append(doc_graph)
            except Exception as e:
                logger.error('Error processing doc: %s', str(e))
                logger.error('Traceback: %s', traceback.format_exc())
                doc_graphs.append(nx.DiGraph())
        return doc_graphs

    def _build_dummy_graph(self) -> nx.Graph:
        G = nx.DiGraph() if self.graph_type == 'directed' else nx.Graph()
        # Add minimal dummy structure
        G.add_node("DUMMY_NOUN", pos_tag="NOUN")
        G.add_node("FILLER_VERB", pos_tag="VERB")
        G.add_edge("DUMMY_NOUN", "FILLER_VERB", gramm_relation="dep", token_distance=1.0)
        return G

    def transform(self, nlp_docs, set) -> list:
        doc_graphs = self._transform_pipeline(nlp_docs, set)
        output_list = []
        avg_nodes = 0
        avg_edges = 0

        for i, graph in enumerate(doc_graphs):
            # Fallback for malformed or short graphs
            if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
                logger.warning(f"[ISG] Using fallback graph for doc_id={i} (nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()})")
                graph = self._build_dummy_graph()

            output_list.append({
                'doc_id': i,
                'graph': graph,
                'number_of_edges': graph.number_of_edges(),
                'number_of_nodes': graph.number_of_nodes(),
                'status': 'success'
            })

            avg_nodes += graph.number_of_nodes()
            avg_edges += graph.number_of_edges()

        return output_list, avg_nodes / len(output_list), avg_edges / len(output_list)


class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dense_hidden_dim,
        output_dim,
        dropout,
        num_layers,
        edge_attr=False,
        gnn_type='GCNConv',
        heads=1,
        task='node',
        norm_type='batchnorm',   # 'batchnorm', 'layernorm', or None
        post_mp_layers=3,        # number of layers after message passing
        pooling_type='mean'      # 'mean', 'max', 'sum', 'attention', 'set2set'
    ):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.edge_attr = edge_attr
        self.dropout = dropout
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.pooling_type = pooling_type
        self.post_mp_layers = post_mp_layers
        self.dense_hidden_dim = dense_hidden_dim

        # First conv
        self.conv1 = self.build_conv_model(input_dim, hidden_dim, heads)
        self.norm1 = self.build_norm_layer(hidden_dim * heads)

        # Additional conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self.build_conv_model(hidden_dim * heads, hidden_dim, heads))
            self.norms.append(self.build_norm_layer(hidden_dim * heads))

        # Global Pooling
        self.global_pool = self.get_pooling_layer(pooling_type, hidden_dim * heads)

        # Post-message-passing MLP
        dims = [hidden_dim * heads] + [dense_hidden_dim // (2 ** i) for i in range(post_mp_layers - 1)] + [output_dim]
        post_mp = []
        for i in range(len(dims) - 1):
            post_mp.append(nn.Linear(dims[i], dims[i + 1]))
            #if i < len(dims) - 2:
            #    post_mp.append(nn.ReLU())
        self.post_mp = nn.Sequential(*post_mp)

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.gnn_type == 'GCNConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GINConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, heads=heads)
        elif self.gnn_type == 'TransformerConv':
            if self.edge_attr:
                return TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=32)
            else:
                return TransformerConv(input_dim, hidden_dim, heads=heads)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def build_norm_layer(self, dim):
        if self.norm_type == 'batchnorm':
            return nn.BatchNorm1d(dim)
        elif self.norm_type == 'layernrom':
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

    def get_pooling_layer(self, pooling_type, hidden_dim):
        if pooling_type == 'mean':
            return global_mean_pool
        elif pooling_type == 'max':
            return global_max_pool
        elif pooling_type == 'sum':
            return global_add_pool
        elif pooling_type == 'attention':
            gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
            return GlobalAttention(gate_nn)
        elif pooling_type == 'set2set':
            return Set2Set(hidden_dim, processing_steps=3)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")


    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None):
        if self.edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            if self.edge_attr:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.global_pool(x, batch)
        return x

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.get_graph_embedding(x, edge_index, edge_attr, batch)
        logits = self.post_mp(x)
        return logits
        

def load_model(path_weights="gnn_model.pth", path_config="gnn_config.json", device="cpu"):
    with open(path_config, "r") as f:
        config = json.load(f)
    print(config)

    model = GNN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        dense_hidden_dim=config["dense_hidden_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        edge_attr=config["edge_attr"],
        gnn_type=config["gnn_type"],
        heads=config["heads"],
        task=config["task"],
        norm_type=config["norm_type"],
        post_mp_layers=config["post_mp_layers"],
        pooling_type=config["pooling_type"]
    ).to(device)

    model.load_state_dict(torch.load(path_weights, map_location=device))
    model.eval()
    return model

def get_node_features_from_doc(graph, text, tokenizer, model, device, max_length=512):
    # 1. Tokenize entire document
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    graph_nodes = list(graph.nodes())

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]

    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
    token_map = {
        token.lstrip("Ġ▁").lower(): i
        for i, token in enumerate(input_tokens)
    }
    node_features = []

    match_cnt = 0
    no_match_cnt = 0
    cls_embedding = hidden_states[0].detach().cpu()
    for node in graph_nodes:
        word = node.split("_")[0].lower()
        
        if node == "ROOT_0":
            match_cnt += 1
            node_features.append(cls_embedding)
            continue

        idx = token_map.get(word)
        if idx is not None:
            match_cnt += 1
            node_features.append(hidden_states[idx].detach().cpu())
        else:
            no_match_cnt += 1
            node_features.append(torch.zeros(model.config.hidden_size).detach().cpu())

    #print(graph.number_of_nodes(), graph.number_of_edges(), match_cnt, no_match_cnt)
    return torch.stack(node_features)  # shape: [num_nodes, hidden_dim]

def add_graph_attr(graph):
    for u, v, attrs in graph.edges(data=True):
        try:
            tag = attrs.get('gramm_relation', 'dep')
            tag_id = DEP2IDX.get(tag, DEP2IDX['dep'])
            graph[u][v]['dep_id'] = tag_id
            graph[u][v]['token_distance'] = float(attrs.get('token_distance', 1))
        except Exception as e:
            # Log error and assign default values
            logger.warning(f"Missing edge attributes for ({u}, {v}) — using defaults. Error: {e}")
            graph[u][v]['dep_id'] = DEP2IDX['dep']
            graph[u][v]['token_distance'] = 1.0
    return graph

def get_pos_encoding(graph):
    """
    Return POS IDs (not one-hot) for input to nn.Embedding
    """
    graph_nodes = list(graph.nodes())
    pos_ids = []
    for node in graph_nodes:
        if "_" not in node:
            pos = "X"
        else:
            pos = node.split("_")[-1]
        pos_id = POS2IDX.get(pos, POS2IDX["X"])
        pos_ids.append(pos_id)
    return torch.tensor(pos_ids, dtype=torch.long)  # [num_nodes]


def extract_feat(graph_data, texts, device, tokenizer, model, projector,
                 pos_embedding, dep_embedding, reduce_dim_emb=False, add_pos_feat=True,
                 project_after_concat=False, set="train"):
    
    data_list = []
    for g_instance in tqdm(graph_data, desc=f"graph-feat {set}: "):
        try:
            graph = g_instance['graph']
            doc_id = g_instance['doc_id']

            # 1. LLM-based node features
            node_feats = get_node_features_from_doc(graph, texts[doc_id], tokenizer, model, device)

            # 2. Graph-level attributes
            graph = add_graph_attr(graph)

            # 3. POS embeddings (optional)
            if add_pos_feat:
                with torch.no_grad():
                    pos_ids = get_pos_encoding(graph)  # [num_nodes]
                    pos_feats = pos_embedding(pos_ids).cpu()
            else:
                pos_feats = None

            # 5. Feature combination logic
            if reduce_dim_emb:
                if project_after_concat:
                    # Concatenate all, then project
                    features = [node_feats]
                    if pos_feats is not None:
                        features.append(pos_feats)
                    full_input = torch.cat(features, dim=1)
                    with torch.no_grad():
                        node_feats = projector(full_input.to(device)).cpu()
                else:
                    # Project LLM First, then add POS + Domain
                    with torch.no_grad():
                        node_feats = projector(node_feats.to(device)).cpu()
                    features = [node_feats]
                    if pos_feats is not None:
                        features.append(pos_feats)
                    node_feats = torch.cat(features, dim=1)
                    
            else:
                # No projection, just concatenate everything
                features = [node_feats]
                if pos_feats is not None:
                    features.append(pos_feats)
                node_feats = torch.cat(features, dim=1)

            # 6. Convert to PyG
            pyg_data = from_networkx(graph)
            with torch.no_grad():
                dep_embed_tensor = dep_embedding(pyg_data.dep_id.to(device)).cpu()
            pyg_data.edge_attr = dep_embed_tensor
            pyg_data.token_distance = torch.tensor(
                [graph[u][v]['token_distance'] for u, v in graph.edges()],
                dtype=torch.float,
                device=device
            )
            pyg_data.x = node_feats
            data_list.append(pyg_data)

        except Exception as e:
            logger.error(f"[extract_feat] Error in doc_id={doc_id}: {str(e)}")
            #logger.error('Traceback: %s', traceback.format_exc())

    return data_list

def normalize_text(texts, special_chars=False, stop_words=False, set='train'):    
    all_texts_norm = []
    for text in tqdm(texts, desc=f"Normalizing {set} corpus"):
        text_norm = utils.text_normalize(text, special_chars, stop_words) 
        all_texts_norm.append(text_norm)
    return all_texts_norm

def nlp_pipeline(docs: list, set):
    doc_lst = []
    #for doc in list(nlp.pipe(docs, as_tuples=False, n_process=4, batch_size=128)):
    for nlp_doc in tqdm(nlp.pipe(docs, batch_size=64, n_process=4), total=len(docs), desc=f"nlp_spacy_docs for {set} set"):

        doc_lst.append(nlp_doc)
    return doc_lst

def build_projector(input_dim=768, hidden_dim=512, output_dim=256, dropout=0.1, num_layers=2, use_norm=True, activation='relu'):
    layers = []
    dim_in = input_dim
 
    for i in range(num_layers):
        dim_out = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(dim_in, dim_out))

        if i < num_layers - 1:
            if use_norm:
                layers.append(nn.BatchNorm1d(dim_out))  # or nn.BatchNorm1d
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(dropout))

        dim_in = dim_out
    return nn.Sequential(*layers)

def test_gnn(model, device, loader, criterion):
    model.eval()
    all_preds = []
    all_probs = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            #out = model(data)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_probs.extend(out.cpu().numpy())

    return all_preds, all_probs

def extract_graph_feat(
        set_texts, project_after_concat, add_pos_feat, graph_type,
        input_proj_dim, pos_emb_dim, dep_emb_dim, reduced_dim, reduce_dim_emb, 
        tokenizer, lang_model, file_name_data, output_dir, device, partition, save_data=True
    ):
    
    set_texts_norm = normalize_text(set_texts, set=partition)
    set_nlp_docs = nlp_pipeline(set_texts_norm, set=partition)

    isg = ISG(graph_type=graph_type, output_format='networkx', apply_prep=True, parallel_exec=False, language='en')
    graph_data, _, _= isg.transform(set_nlp_docs, set=partition)

    if project_after_concat and add_pos_feat:
        input_proj_dim += pos_emb_dim

    #llm_projector = nn.Linear(input_proj_dim, reduced_dim).to(device)
    llm_projector = build_projector(input_dim=input_proj_dim, hidden_dim=512, output_dim=reduced_dim, dropout=0.1, num_layers=1, use_norm=False, activation='gelu').to(device)
    pos_embedding = nn.Embedding(NUM_POS_TAGS, pos_emb_dim)
    dep_embedding = nn.Embedding(NUM_DEP_TAGS, dep_emb_dim).to(device)

    data_list = extract_feat(graph_data, set_texts, device, tokenizer, 
                            lang_model, llm_projector, pos_embedding, dep_embedding, 
                            reduce_dim_emb, add_pos_feat, project_after_concat, set=partition)
    data = {
        "graph_data": graph_data,
        "data_list": data_list
    }

    if save_data:
        file_name_data = f'{file_name_data}_{partition}'
        utils.save_data(data, file_name_data, path=f'{output_dir}/', format_file='.pkl', compress=False)
    return graph_data, data_list


def test_inference(test_text, model_save_path, model_config_path,
                   tokenizer, lang_model, output_dir, device, batch_size=32, 
                   project_after_concat=True, add_pos_feat=True, 
                   graph_type='isg', input_proj_dim=256, pos_emb_dim=32, dep_emb_dim=32,
                   reduced_dim=256, reduce_dim_emb=True):
    # === Load model using your existing helper ===
    model = load_model(path_weights=model_save_path, path_config=model_config_path, device=device)
    print(model)
    file_name_data=''

    graph_test, data_test_list = extract_graph_feat(
        test_text,
        project_after_concat, add_pos_feat, graph_type,
        input_proj_dim, pos_emb_dim, dep_emb_dim,
        reduced_dim, reduce_dim_emb,
        tokenizer, lang_model,
        file_name_data, output_dir, device,
        partition='test', save_data=False
    )

    test_loader = DataLoader(data_test_list, batch_size=batch_size, shuffle=False, num_workers=0)

    # === Run inference ===
    criterion = torch.nn.CrossEntropyLoss()
    preds_test, probs_test = test_gnn(model, device, test_loader, criterion)
    #print("preds_test: ", preds_test)
    #print("probs_test: ", probs_test)
    return preds_test, probs_test


def main(args):
    
    cuda_num = 0
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    #lang_model_name =  'microsoft/deberta-v3-base'
    lang_model_name  = "/root/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
    tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
    lang_model = AutoModel.from_pretrained(lang_model_name).to(device) 

    test_set = utils.read_jsonl(dir_path=args.input_dataset)
    test_text = list(test_set['text']) 

    # ****tmp
    #cut_off_dataset = "X_X_1"
    #cut_off_test = int(cut_off_dataset.split('_')[2])
    #test_text = test_text[:int(len(test_text) * (cut_off_test / 100))]
    # ****tmp

    prefix_model = 'isg_data_pan25_10_10_10perc'
    preds_test, logits_gnn = test_inference(
        test_text,
        model_save_path=f"outputs/gnn_model_{prefix_model}.pth",
        model_config_path=f"outputs/config_{prefix_model}.json",
        tokenizer=tokenizer,
        lang_model=lang_model,
        output_dir="outputs", 
        device=device,
        batch_size=64, 
        project_after_concat=True, 
        add_pos_feat=True, 
        input_proj_dim=lang_model.config.hidden_size, 
        pos_emb_dim=32, 
        dep_emb_dim=32,
        reduce_dim_emb=False,
        reduced_dim=256, 
    )

    test_set = test_set.to_dict('records')
    predictions = []
    epsilon = 0.05  # undecidable margin

    for row, logits in zip(test_set, logits_gnn):
        # Convert logits to probabilities
        logits_tensor = torch.tensor(logits)
        probs = F.softmax(logits_tensor, dim=0).numpy()

        prob_machine = float(probs[1])  # Probability of class 1 (machine)
        pred_class = int(np.argmax(probs))

        # Apply undecidable rule
        if abs(prob_machine - 0.5) <= epsilon:
            label = 0.5
        else:
            label = round(prob_machine, 4) 

        predictions.append({"id": row["id"], "label": label})
    
    #print(predictions)
    utils.save_json2(data=predictions, file_path=args.output_dir + '/' + str(int(time.time())) + "preds.jsonl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Positional arguments (no --prefix)
    parser.add_argument("input_dataset", help="Path to input dataset (e.g., data.json)")
    parser.add_argument("output_dir", help="Directory to save prediction outputs", default=utils.OUTPUT_DIR_PATH, type=str)

    # Optional named arguments
    parser.add_argument("--exec_type", choices=["train", "test"], default="test", help="Execution type")
    parser.add_argument("--model", help="Model to train or test", default="SGDClassifier", type=str)

    args = parser.parse_args()
    main(args)