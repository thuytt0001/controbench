import torch
import dgl
import json
import numpy as np
from collections import Counter
import random

# Set all random seeds for deterministic behavior
def set_deterministic_behavior(seed=42):
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Force single-threading
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set DGL's random seed
    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except:
        print("Note: DGL version does not support direct random seed setting")
    
    # Set environment variables
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def load_split_graph_data(train_file, test_file=None):
    """
    Load pre-split train and test graph data from JSON files.
    
    Args:
        train_file: Path to training data JSON file
        test_file: Path to test data JSON file (optional)
        
    Returns:
        train_graph: DGL graph for training
        test_graph: DGL graph for testing (if test_file provided)
        train_labels: Node labels for training
        test_labels: Node labels for testing (if test_file provided)
        flair_to_idx: Mapping from node labels to indices
        idx_to_flair: Mapping from indices to node labels
    """
    # Set deterministic behavior first
    set_deterministic_behavior(42)

    # Load training data
    with open(train_file, "r") as f:
        train_data = json.load(f)
    
    # Process training data
    train_graph, train_user_nodes, train_user_map, train_labels, flair_to_idx = process_graph_data(train_data)
    
    if test_file:
        # Load test data
        with open(test_file, "r") as f:
            test_data = json.load(f)
        
        # Process test data (use same flair_to_idx mapping)
        test_graph, test_user_nodes, test_user_map, test_labels, _ = process_graph_data(
            test_data, existing_flair_map=flair_to_idx)
        
        idx_to_flair = {i: flair for flair, i in flair_to_idx.items()}
        
        return (train_graph, test_graph, train_labels, test_labels, 
                flair_to_idx, idx_to_flair, train_user_map, test_user_map)
    else:
        idx_to_flair = {i: flair for flair, i in flair_to_idx.items()}
        return train_graph, train_labels, flair_to_idx, idx_to_flair, train_user_map

def process_graph_data(data, existing_flair_map=None):
    """
    Process JSON graph data into DGL format.
    
    Args:
        data: JSON data with nodes and edges
        existing_flair_map: Existing mapping from flairs to indices (optional)
        
    Returns:
        dgl_graph: DGL heterogeneous graph
        user_nodes: Dictionary of user nodes
        user_map: Mapping from user IDs to indices
        labels: Node labels
        flair_to_idx: Mapping from node labels to indices
    """
    nodes = data["nodes"]
    edges = data["edges"]
    
    # Separate user and post nodes
    user_nodes = {node["id"]: node for node in nodes if node["type"] == "user"}
    post_nodes = {node["id"]: node for node in nodes if node["type"] == "post"}
    
    num_users = len(user_nodes)
    num_posts = len(post_nodes)
    
    print(f"Detected {num_users} users and {num_posts} posts in {data.get('split', 'unknown')} set")
    
    # Assign unique numeric IDs
    user_map = {user_id: i for i, user_id in enumerate(user_nodes)}
    post_map = {post_id: i for i, post_id in enumerate(post_nodes)}
    
    # Process edges
    publish_edges, comment_edges, user_comment_user_edges = [], [], []
    comment_features, user_comment_user_features, ucu_reply_features = [], [], []
    
    for edge in edges:
        src, dst, edge_type = edge["source"], edge["target"], edge["type"]
        
        if edge_type == "user_publish_post":
            if src in user_map and dst in post_map:
                publish_edges.append((user_map[src], post_map[dst]))
    
        elif edge_type == "user_comment_post":
            if src in user_map and dst in post_map:
                comment_edges.append((user_map[src], post_map[dst]))
                if "embedding" in edge:
                    comment_features.append(torch.tensor(edge["embedding"]))
    
        elif edge_type == "user_comment_user":
            if src in user_map and dst in user_map:
                user_comment_user_edges.append((user_map[src], user_map[dst]))
                # "embedding"       = User A's original comment (what was said to User B)
                # "reply_embedding" = User B's reply            (direct stance signal)
                if "embedding" in edge:
                    user_comment_user_features.append(
                        torch.tensor(edge["embedding"], dtype=torch.float))
                if "reply_embedding" in edge:
                    ucu_reply_features.append(
                        torch.tensor(edge["reply_embedding"], dtype=torch.float))
    
    # Convert to DGL graph format
    dgl_graph = dgl.heterograph({
        ("user", "publish", "post"): publish_edges,
        ("user", "comment", "post"): comment_edges,
        ("user", "user_comment_user", "user"): user_comment_user_edges
    })
    
    # Default embedding dimension - BERT base produces 768-dimensional embeddings
    default_embedding_dim = 768
    
    # Extract post embeddings - with special handling for title and content embeddings
    post_features = []
    post_embedding_dim = default_embedding_dim
    
    for post in post_nodes.values():
        if "embedding" in post and "title_embedding" in post:
            # If both are available, concatenate them
            content_feat = torch.tensor(post["embedding"], dtype=torch.float)
            title_feat = torch.tensor(post["title_embedding"], dtype=torch.float)
            
            # Check dimensions before combining
            if len(post_features) == 0:  # Just log once
                print(f"Found both title and content embeddings. Content dim: {content_feat.shape}, Title dim: {title_feat.shape}")
            
            # Concatenate features
            combined_feat = torch.cat([title_feat, content_feat], dim=0)
            post_features.append(combined_feat)
            
            # Update the post embedding dimension if needed
            if combined_feat.shape[0] > post_embedding_dim:
                post_embedding_dim = combined_feat.shape[0]
                
        elif "embedding" in post:
            # Only content embedding available
            post_feat = torch.tensor(post["embedding"], dtype=torch.float)
            post_features.append(post_feat)
            # Check if we need to update our embedding dimension
            if post_feat.shape[0] > post_embedding_dim:
                post_embedding_dim = post_feat.shape[0]
                
        elif "title_embedding" in post:
            # Only title embedding available
            post_feat = torch.tensor(post["title_embedding"], dtype=torch.float)
            post_features.append(post_feat)
            if post_feat.shape[0] > post_embedding_dim:
                post_embedding_dim = post_feat.shape[0]
                
        else:
            # Create zero tensor with default dimension
            post_features.append(torch.zeros(default_embedding_dim))
    
    # Convert post features to tensor
    if post_features:
        # Make sure all post features have the same dimension
        for i, feat in enumerate(post_features):
            if feat.shape[0] < post_embedding_dim:
                # Pad with zeros if needed
                padded = torch.zeros(post_embedding_dim)
                padded[:feat.shape[0]] = feat
                post_features[i] = padded
                
        post_features = torch.stack(post_features)
    else:
        post_features = torch.zeros((len(post_nodes), default_embedding_dim))
    
    # Log the actual dimensions
    print(f"Post feature dimension: {post_features.shape[1]}")
    
    # User features are empty (DGL requires at least a placeholder tensor)
    # Set to same dimension as post features for consistency
    user_features = torch.zeros((len(user_nodes), default_embedding_dim))
    
    # Assign node features separately for each node type
    dgl_graph.ndata["feat"] = {
        "user": user_features,
        "post": post_features
    }
    
    # Convert comment features to tensor
    if comment_features:
        # Check dimensions
        comment_embedding_dim = max(feat.shape[0] for feat in comment_features)
        print(f"Comment edge feature dimension: {comment_embedding_dim}")
        
        # Make sure all comment features have the same dimension
        for i, feat in enumerate(comment_features):
            if feat.shape[0] < comment_embedding_dim:
                padded = torch.zeros(comment_embedding_dim)
                padded[:feat.shape[0]] = feat
                comment_features[i] = padded
        
        comment_features = torch.stack(comment_features)
        dgl_graph.edges["comment"].data["feat"] = comment_features
    
    if user_comment_user_features:
        # Check dimensions
        ucu_embedding_dim = max(feat.shape[0] for feat in user_comment_user_features)
        print(f"User-comment-user edge feature dimension: {ucu_embedding_dim}")
        
        # Make sure all user_comment_user features have the same dimension
        for i, feat in enumerate(user_comment_user_features):
            if feat.shape[0] < ucu_embedding_dim:
                padded = torch.zeros(ucu_embedding_dim)
                padded[:feat.shape[0]] = feat
                user_comment_user_features[i] = padded
        
        user_comment_user_features = torch.stack(user_comment_user_features)
        dgl_graph.edges["user_comment_user"].data["feat"] = user_comment_user_features

    # Load User B's reply embedding as "reply_feat"
    if ucu_reply_features:
        ucu_reply_dim = max(feat.shape[0] for feat in ucu_reply_features)
        print(f"Parent content feature dimension: {ucu_reply_dim}")

        for i, feat in enumerate(ucu_reply_features):
            if feat.shape[0] < ucu_reply_dim:
                padded = torch.zeros(ucu_reply_dim)
                padded[:feat.shape[0]] = feat
                ucu_reply_features[i] = padded

        ucu_reply_features = torch.stack(ucu_reply_features)
        # Only attach if same number of edges as UCU edges
        if ucu_reply_features.shape[0] == dgl_graph.num_edges("user_comment_user"):
            dgl_graph.edges["user_comment_user"].data["reply_feat"] = ucu_reply_features
            print(f"Added reply features for {ucu_reply_features.shape[0]} "
                  f"user-comment-user edges")
    
    # Create labels for user nodes
    if existing_flair_map is None:
        flairs = sorted(set(node["label"] for node in user_nodes.values()))
        flair_to_idx = {flair: i for i, flair in enumerate(flairs)}
    else:
        flair_to_idx = existing_flair_map
    
    labels = torch.full((dgl_graph.num_nodes('user'),), -1, dtype=torch.long)
    
    for user_id, node in user_nodes.items():
        user_idx = user_map[user_id]
        if node["label"] in flair_to_idx:
            labels[user_idx] = flair_to_idx[node["label"]]
    
    # Count class distribution
    class_counts = torch.bincount(labels[labels != -1])
    print(f"Class distribution: {class_counts.tolist()}")
    
    return dgl_graph, user_nodes, user_map, labels, flair_to_idx