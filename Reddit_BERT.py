import torch
from transformers import BertTokenizer, BertModel
import json

# Initialize BERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)
model.eval()

# 1. Change the route and name of json file
with open('data/graph_data_trump.json', 'r') as f:  # Change this to your JSON file
    data = json.load(f)

# Extract text content
post_titles = []
post_contents = []
user_comment_post_features = []
user_comment_user_features = []
reply_content_features = []

# To avoid duplicate processing, keep track of unique texts
unique_texts = set()
text_to_category = {}

# Extract post content and titles
for node in data["nodes"]:
    if node["type"] == "post":
        title = str(node.get("title", ""))
        content = str(node.get("content", ""))
        
        post_titles.append(title)
        post_contents.append(content)
        
        # Track unique texts
        if title and title not in unique_texts:
            unique_texts.add(title)
            text_to_category[title] = "post_title"
        if content and content not in unique_texts:
            unique_texts.add(content)
            text_to_category[content] = "post_content"

# Extract edge features
for edge in data["edges"]:
    if edge["type"] == "user_comment_post":
        content = str(edge.get("content", ""))
        user_comment_post_features.append(content)
        
        if content and content not in unique_texts:
            unique_texts.add(content)
            text_to_category[content] = "user_comment_post"
            
    elif edge["type"] == "user_comment_user":
        content = str(edge.get("content", ""))
        reply_content = str(edge.get("reply_content", ""))
        
        user_comment_user_features.append(content)
        reply_content_features.append(reply_content)
        
        if content and content not in unique_texts:
            unique_texts.add(content)
            text_to_category[content] = "user_comment_user"
        if reply_content and reply_content not in unique_texts:
            unique_texts.add(reply_content)
            text_to_category[reply_content] = "reply_content"

print(f"Found {len(post_titles)} posts")
print(f"Found {len(user_comment_post_features)} user-comment-post edges")
print(f"Found {len(user_comment_user_features)} user-comment-user edges")
print(f"Found {len(reply_content_features)} reply contents")
print(f"Total unique texts to process: {len(unique_texts)}")

def mean_embedding(text_list):
    if not text_list:
        return torch.empty(0, 768)
    
    embeddings = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        embeddings.append(pooled_output.cpu())
    
    return torch.cat(embeddings, dim=0)

# Generate embeddings for unique texts first (to avoid duplicates)
print("Generating embeddings for unique texts...")
unique_text_list = list(unique_texts)
unique_embeddings = mean_embedding(unique_text_list)

# Create a mapping from text to embedding
text_to_embedding = {}
for i, text in enumerate(unique_text_list):
    text_to_embedding[text] = unique_embeddings[i]

print("Organizing embeddings by category...")

# Organize embeddings by original lists
mean_embedding_post_title = []
for title in post_titles:
    if title in text_to_embedding:
        mean_embedding_post_title.append(text_to_embedding[title])
    else:
        mean_embedding_post_title.append(torch.zeros(768))

mean_embedding_post_content = []
for content in post_contents:
    if content in text_to_embedding:
        mean_embedding_post_content.append(text_to_embedding[content])
    else:
        mean_embedding_post_content.append(torch.zeros(768))

mean_embedding_user_comment_post = []
for content in user_comment_post_features:
    if content in text_to_embedding:
        mean_embedding_user_comment_post.append(text_to_embedding[content])
    else:
        mean_embedding_user_comment_post.append(torch.zeros(768))

mean_embedding_user_comment_user = []
for content in user_comment_user_features:
    if content in text_to_embedding:
        mean_embedding_user_comment_user.append(text_to_embedding[content])
    else:
        mean_embedding_user_comment_user.append(torch.zeros(768))

mean_embedding_reply_content = []
for reply in reply_content_features:
    if reply in text_to_embedding:
        mean_embedding_reply_content.append(text_to_embedding[reply])
    else:
        mean_embedding_reply_content.append(torch.zeros(768))

# Convert lists to tensors
mean_embedding_post_title = torch.stack(mean_embedding_post_title) if mean_embedding_post_title else torch.empty(0, 768)
mean_embedding_post_content = torch.stack(mean_embedding_post_content) if mean_embedding_post_content else torch.empty(0, 768)
mean_embedding_user_comment_post = torch.stack(mean_embedding_user_comment_post) if mean_embedding_user_comment_post else torch.empty(0, 768)
mean_embedding_user_comment_user = torch.stack(mean_embedding_user_comment_user) if mean_embedding_user_comment_user else torch.empty(0, 768)
mean_embedding_reply_content = torch.stack(mean_embedding_reply_content) if mean_embedding_reply_content else torch.empty(0, 768)

# Save embeddings
embeddings = {
    'post_title': mean_embedding_post_title,
    'post_content': mean_embedding_post_content,
    'user_comment_post': mean_embedding_user_comment_post,
    'user_comment_user': mean_embedding_user_comment_user,
    'reply_content': mean_embedding_reply_content,
    'text_to_embedding_map': text_to_embedding  # For easy lookup later
}

# 2. Change the torch name
torch.save(embeddings, 'trump_embedding.pt')  # Change this filename

print("Embeddings saved!")
print(f"Post titles: {mean_embedding_post_title.shape}")
print(f"Post contents: {mean_embedding_post_content.shape}")
print(f"User-comment-post: {mean_embedding_user_comment_post.shape}")
print(f"User-comment-user: {mean_embedding_user_comment_user.shape}")
print(f"reply contents: {mean_embedding_reply_content.shape}")
print(f"Total unique texts processed: {len(text_to_embedding)}")
print("Finished!")