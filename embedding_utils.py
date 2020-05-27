import numpy as np
import json
# util functions for loading embeddings

# compute the embedding from a text
def get_embedding_from_text(text, text_embedding_dict, sentence_bert_model):
    if not text_embedding_dict is None and text in text_embedding_dict:
        return text_embedding_dict[text]
    else:
        return sentence_bert_model.encode([text])[0]

# compute the embeddings from a list of texts
def get_embeddings_from_texts(texts, text_embedding_dict, sentence_bert_model):
    embeddings = sentence_bert_model.encode(texts)
    return dict(zip(texts, embeddings))

# compute the aggregated (average) embedding from a list of texts
def get_an_aggregated_embedding_from_texts(texts, text_embedding_dict, sentence_bert_model):
    agg_embedding = None
    all_text_found = True
    if texts is None or len(texts) == 0:
        return None
    # try using the text_embedding_dict first
    if not text_embedding_dict is None:
        count = 0
        for text in texts:
            if not text in text_embedding_dict:
                all_text_found = False
                break
            embedding = text_embedding_dict[text]
            if agg_embedding is None:
                agg_embedding = embedding
                count += 1
            else:
                agg_embedding = np.add(agg_embedding, embedding)
                count += 1
        if all_text_found and isinstance(agg_embedding, np.ndarray):
            return np.divide(agg_embedding, count)

    # calculate the embeddings if all_text_found is False
    embeddings = sentence_bert_model.encode(texts)
    if len(embeddings) > 0:
        agg_embedding = embeddings[0]
        for embedding in embeddings[1:]:
            agg_embedding = np.add(agg_embedding, embedding)
        return np.divide(agg_embedding, len(embeddings))
    else:
        return None





