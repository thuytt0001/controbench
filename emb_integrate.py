# LOAD EMBEDDINGS
import torch
import json

# Load the embeddings
embedding_data = torch.load('religion_embedding.pt')  # Change filename
post_title_embeddings = embedding_data['post_title']
post_content_embeddings = embedding_data['post_content']
user_comment_post_embeddings = embedding_data['user_comment_post']
user_comment_user_embeddings = embedding_data['user_comment_user']
reply_content_embeddings = embedding_data['parent_content']
text_to_embedding_map = embedding_data.get('text_to_embedding_map', {})

print(f"Loaded embeddings:")
print(f"Post titles: {post_title_embeddings.shape}")
print(f"Post contents: {post_content_embeddings.shape}")
print(f"User-comment-post: {user_comment_post_embeddings.shape}")
print(f"User-comment-user: {user_comment_user_embeddings.shape}")
print(f"reply contents: {reply_content_embeddings.shape}")
print(f"Text-to-embedding map size: {len(text_to_embedding_map)}")

#------------------------------------------------------------------------------------------------
# ADD EMBEDDINGS TO THE GRAPH
# Load the original graph data
with open('data/graph_data_religion.json', 'r') as f:  # Change filename
    graph_data = json.load(f)

# Create a new graph structure with embeddings
graph_with_embeddings = {
    "nodes": [],
    "edges": []
}

# Add embeddings to nodes
post_counter = 0
for node in graph_data["nodes"]:
    new_node = node.copy()
    if node["type"] == "post":
        title = str(node.get("title", ""))
        content = str(node.get("content", ""))
        
        # Use direct text lookup instead of counters
        if content in text_to_embedding_map:
            new_node["embedding"] = text_to_embedding_map[content].tolist()
        elif post_counter < len(post_content_embeddings):
            # Fallback to counter method
            new_node["embedding"] = post_content_embeddings[post_counter].tolist()
        
        if title in text_to_embedding_map:
            new_node["title_embedding"] = text_to_embedding_map[title].tolist()
        elif post_counter < len(post_title_embeddings):
            # Fallback to counter method
            new_node["title_embedding"] = post_title_embeddings[post_counter].tolist()
            
        post_counter += 1
    graph_with_embeddings["nodes"].append(new_node)

print(f"Processed {post_counter} post nodes")

# Add embeddings to edges using direct text lookup
user_comment_post_attached = 0
user_comment_user_attached = 0
user_comment_post_counter = 0  # Keep counters for fallback
user_comment_user_counter = 0

for edge in graph_data["edges"]:
    new_edge = edge.copy()
    
    if edge["type"] == "user_comment_post":
        content = str(edge.get("content", ""))
        
        # Primary method: Direct text lookup
        if content and content in text_to_embedding_map:
            new_edge["embedding"] = text_to_embedding_map[content].tolist()
            user_comment_post_attached += 1
        # Fallback method: Counter-based
        elif user_comment_post_counter < len(user_comment_post_embeddings):
            new_edge["embedding"] = user_comment_post_embeddings[user_comment_post_counter].tolist()
            user_comment_post_attached += 1
        
        user_comment_post_counter += 1
    
    elif edge["type"] == "user_comment_user":
        content = str(edge.get("content", ""))
        reply_content = str(edge.get("parent_content", ""))
        
        # Primary method: Direct text lookup for content
        if content and content in text_to_embedding_map:
            new_edge["embedding"] = text_to_embedding_map[content].tolist()
            user_comment_user_attached += 1
        # Fallback method: Counter-based for content
        elif user_comment_user_counter < len(user_comment_user_embeddings):
            new_edge["embedding"] = user_comment_user_embeddings[user_comment_user_counter].tolist()
            user_comment_user_attached += 1
        
        # Primary method: Direct text lookup for reply content
        if reply_content and reply_content in text_to_embedding_map:
            new_edge["reply_embedding"] = text_to_embedding_map[reply_content].tolist()
        # Fallback method: Counter-based for reply content
        elif user_comment_user_counter < len(reply_content_embeddings):
            new_edge["reply_embedding"] = reply_content_embeddings[user_comment_user_counter].tolist()
        
        user_comment_user_counter += 1
    
    graph_with_embeddings["edges"].append(new_edge)

print(f"Successfully attached embeddings:")
print(f"- {user_comment_post_attached} user-comment-post edges (out of {user_comment_post_counter} total)")
print(f"- {user_comment_user_attached} user-comment-user edges (out of {user_comment_user_counter} total)")

# Count reply embeddings attached
reply_embedidngs_attached = sum(1 for edge in graph_with_embeddings["edges"] 
                                if edge.get("type") == "user_comment_user" and "reply_embedding" in edge)
print(f"- {reply_embedidngs_attached} reply content embeddings attached")

# Save the graph with embeddings
new_file_name = 'religion_graph_data_with_embeddings.json'  # Change filename
with open(new_file_name, 'w') as f:
    json.dump(graph_with_embeddings, f)

print(f"Graph with embeddings saved to '{new_file_name}'")

#-------------------------------------------------------------------------------------------
# Optional: Save embeddings in separate files for easy loading in GNN training
torch.save({
    'node_features': {
        'user': torch.zeros(len([n for n in graph_data['nodes'] if n['type'] == 'user']), 768),  # Users have no text features
        'post': post_content_embeddings
    },
    'edge_features': {
        'user_comment_post': user_comment_post_embeddings,
        'user_comment_user': user_comment_user_embeddings,
        'reply_content': reply_content_embeddings  # For user-comment-user reply references
    },
    'text_lookup': text_to_embedding_map  # For looking up any text embedding
}, 'religion_features_for_gnn.pt')  # Change filename

print("Features for GNN training saved!")
print("reply embeddings are now included in both the JSON and GNN features!")