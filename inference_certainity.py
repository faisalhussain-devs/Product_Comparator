import json, re
from pathlib import Path
from collections import defaultdict
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import csv
import pandas as pd

aspect_reliability = np.array([
    0.8928571428571428,   # Sub aspect 0
    0.8651162790697675,   # Sub aspect 1
    0.36113549897570973,  # Sub aspect 2
    0.6940559440559441,   # Sub aspect 3
    0.6666666666666667,   # Sub aspect 4
    0.7956043956043957,   # Sub aspect 5
    0.6553846153846155,   # Sub aspect 6
    0.8131609870740306,   # Sub aspect 7
    0.7065546218487395,   # Sub aspect 8
    0.7850045167118338,   # Sub aspect 9
    0.788888888888889,    # Sub aspect 10
    0.8235294117647058,   # Sub aspect 11
    0.5662285136501517,   # Sub aspect 12
    0.4878048780487805,   # Sub aspect 13
    0.6631016042780749,   # Sub aspect 14
    0.7664219838132882,   # Sub aspect 15
    0.7384510869565217,   # Sub aspect 16
    0.6359975594874924,   # Sub aspect 17
    0.7638773819386909,   # Sub aspect 18
    0.7453310696095077,   # Sub aspect 19
    0.9561904761904761,   # Sub aspect 20
    0.7852925389157273,   # Sub aspect 21
    0.46938775510204084,  # Sub aspect 22
    0.8106591865357644,   # Sub aspect 23
    0.6607142857142857,   # Sub aspect 24
    0.747520629873571,    # Sub aspect 25
    0.8133848133848134,   # Sub aspect 26
    0.753880266075388,    # Sub aspect 27
    0.6421311139914045    # Sub aspect 28
])

top_to_specs = {
  0 : [  # battery & charging
    "Battery_Charging",
    "Battery_Type",
    "Our_Tests_Battery",
    "Our_Tests_Battery_(old)"
  ],

  1 : [  # build quality & design
    "Body_Dimensions",
    "Body_Build",
    "Body_Weight",
    "Misc_Colors",
    "Misc_Models"
  ],

  2 : [  # camera
    "Main_Camera_Single",
    "Main_Camera_Dual",
    "Main_Camera_Triple",
    "Main_Camera_Quad",
    "Main_Camera_Video",
    "Main_Camera_Features",
    "Selfie_camera_Single",
    "Selfie_camera_Dual",
    "Selfie_camera_Video",
    "Selfie_camera_Features",
    "Our_Tests_Camera"
  ],

  3 : [  # customer experience
    "Launch_Announced",
  ],

  4 : [  # device features & functionality
    "Features_Sensors",
    "Sound_3.5mm_jack",
    "Comms_Infrared_port",
    "Memory_Card_slot",
    "Sound_Loudspeaker",
    "Our_Tests_Loudspeaker",
    "Network_Technology",
    "Network_GPRS",
    "Network_EDGE"
  ],

  5 : [  # device performance
    "Platform_CPU",
    "Platform_GPU",
    "Memory_Internal",
    "Our_Tests_Performance"
  ],

  6 : [  # display
    "Display_Type",
    "Display_Size",
    "Display_Resolution",
    "Display_Protection",
    "Our_Tests_Display"
  ],

  7 : [  # price & value
       ],

  8 : [  # software & ux
    "Platform_OS"
  ]
}

def clean_value(value: str) -> str:
    """Clean and normalize string values in one go."""
    if not isinstance(value, str):
        return value

    value = re.sub(r'https?://\S+|www\.\S+', '', value)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    value = emoji_pattern.sub('', value)
    value = re.sub(r'\"|\'', '', value)
    value = re.sub(r'\\?/', ',', value)
    value = re.sub(r'(\\n|\n)+', ', ', value)
    value = re.sub(r'\s*,\s*', ', ', value)
    value = re.sub(r',\s*,+', ', ', value)
    value = re.sub(r'^[^a-zA-Z0-9]+', '', value)
    value = re.sub(r'\s+', ' ', value).strip()

    return value

def preprocess_reddit_reviews(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    products = []
    for product in data:
        texts = []
        reviews = product.get("review_texts", [])
        comments = product.get("comments", [])
        for c in comments:
            if isinstance(c, dict):
                body = c.get("body")
                ups = c.get("ups")
                if isinstance(body, str):
                    text = clean_value(body.strip())
                    if len(text.split()) > 15:
                        texts.append({"text": text, "likes": ups if isinstance(ups, int) else 0})

        for review in reviews:
            if isinstance(review, str):
                text = clean_value(review.strip())
                if len(text.split()) > 15:
                    texts.append({"text": text})
        if len(texts) > 0:
          products.append(texts)

    return products

input_path = "/content/Comparator.reviews.json"
product_data = [preprocess_reddit_reviews(input_path)[13]]

def drop_specification_keys(data, drop_keys):
    """Remove specified keys from the 'specifications' dictionary of each item."""
    specs = data.get("specifications")
    if isinstance(specs, dict):
        for key in drop_keys:
            specs.pop(key, None)
    return data


def flatten_data(product, sep="_"):
    """
    Flatten product specifications:
    - Remove everything except 'More Specifications'
    - Rename 'More Specifications' → 'specifications'
    - Flatten title/data pairs recursively
    Clean nested 'reviews' and 'comments' structures:
      - 'reviews' → list of review_text strings
      - 'comments' → two lists: comment bodies & ups
    """

    flat = {}

    def process_specs(specs, parent=""):
        """Recursively flatten title/data pairs into key-value mapping."""
        for entry in specs:
            title = entry.get("title", "").strip().replace(" ", "_")
            data = entry.get("data")

            if isinstance(data, list) and all(isinstance(x, dict) and "title" in x and "data" in x for x in data):
                process_specs(data, parent=f"{parent}{sep}{title}")
            else:
                key = f"{parent}{sep}{title}" if parent else title
                if isinstance(data, list):
                    value = " ".join(str(x).strip() for x in data)
                else:
                    value = data
                flat[key.strip("_")] = value

    if "specification" in product and isinstance(product["specification"], dict):

        more_specs = product["specification"].get("more_specification", [])
        process_specs(more_specs)

    return flat

def preprocess_products(input_file, drop_columns=None):
    specs = {}
    product_name = []
    flat_data = []
    with open(input_file, "r", encoding="utf-8") as f:
      data = json.load(f)
    for product in data:
      product_name.append(product["name"])
      specs["specifications"] = flatten_data(product)
      if drop_columns:
        flat_data.append(drop_specification_keys(specs, drop_keys=drop_columns))
    return flat_data, product_name


DROP_COLUMNS = [
 'Network_2G_bands', 'Network_3G_bands', 'Network_4G_bands',
 'Network_5G_bands', 'Network_Speed', 'Launch_Status', 'Body_SIM', 'Platform_Chipset', 'Comms_WLAN',
 'Comms_Bluetooth', 'Comms_Positioning', 'Comms_NFC', 'Comms_Radio', 'Comms_USB', 'Misc_Price',
 'EU_LABEL_Energy', 'EU_LABEL_Battery', 'EU_LABEL_Free_fall', 'EU_LABEL_Repairability', 'Misc_SAR', 'Misc_SAR_EU'
]

input_path = "/content/Comparator.specifications.json"
data_specs, products_name  = preprocess_products(input_path, DROP_COLUMNS)
product_name = products_name[13]
data_specs = data_specs[13]

def load_tokenizer_and_model_from_hf(mode=0):
    repo_id = "Faisal191/aspect-classifier"
    if mode == 0:
        encoder_subfolder="RankingReviews"
        onnx_path = "RankingReviews/model.onnx"
    elif mode == 1:
        encoder_subfolder = "Domain_trained_encoder"
        onnx_path = "HABSA/Habsa_v6_fp32.onnx"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        subfolder=encoder_subfolder)

    # Download ONNX graph
    onnx_path = hf_hub_download(
        repo_id=repo_id,
        filename=onnx_path,
        repo_type="model")

    return tokenizer, onnx_path

tokenizer, onnx_path = load_tokenizer_and_model_from_hf(mode=0)

providers = [
    ("CUDAExecutionProvider", {
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True}
     )]

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_rank = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
inputs= []
for i, data in enumerate(product_data):
    texts = [review_i["text"] for review_i in data]
    inputs.append(tokenizer(texts, padding=True, truncation=True, max_length=100, return_tensors="np"))

def run_rank_batches(session, inputs, batch_size=128):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    N = len(input_ids)
    outputs = []
    for i in tqdm(range(0, N, batch_size)):
        input_ids_batch = input_ids[i:i+batch_size]
        attention_mask_batch = attention_mask[i:i+batch_size]
        batch_inputs = {
            "input_ids": input_ids_batch.astype(np.int64),
            "attention_mask": attention_mask_batch.astype(np.int64)}
        out = session_rank.run(None, batch_inputs)
        outputs.append(out[0][:,0])
    return outputs

rank_scores = []
for input in inputs:
  rank_score = run_rank_batches(session_rank, input, batch_size=128)
  rank_scores.append(np.concatenate(rank_score))

rank_final = []
product_indx = []
for i, rank_score in enumerate(rank_scores):
  if np.sum(rank_score > 7) > 50:
    rank_final.append(rank_score)
    product_indx.append(i)

tokenizer, onnx_path = load_tokenizer_and_model_from_hf(mode=1)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Enabling aggressive graph optimization
sess_options.enable_profiling = True
session_absa = ort.InferenceSession(
    onnx_path,
    sess_options=sess_options,
    providers=providers)

top_logits_pr, sub_logits_pr, comp_logits_pr, sent_logits_pr = [], [], [], []

inputs = []
for i, data in enumerate(product_data):
  if i in product_indx:
    j = product_indx.index(i)
    indx = rank_final[j] > 5
    input = ([f"Main product: {product_name}. Review: " + data[k]["text"] for k in range(len(data)) if indx[k]])
    inputs.append(tokenizer(input, padding="max_length", truncation=True, max_length=256, return_tensors="np"))

def run_absa_batches(session, inputs, batch_size=300):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    N = len(input_ids)
    outputs = []
    for i in tqdm(range(0, N, batch_size)):
        input_ids_batch = input_ids[i:i+batch_size]
        attention_mask_batch = attention_mask[i:i+batch_size]
        batch_inputs = {
            "input_ids": input_ids_batch.astype(np.int64),
            "attention_mask": attention_mask_batch.astype(np.int64)}
        outputs = session.run(None, batch_inputs)
        top_logits.append(outputs[0])
        sub_logits.append(outputs[1])
        comp_logits.append(outputs[2])
        sent_logits.append(outputs[3])


for input in inputs:
  top_logits, sub_logits, comp_logits, sent_logits = [], [], [], []
  run_absa_batches(session_absa, input, batch_size=128)

  top_logits_pr.append(np.concatenate(top_logits))
  sub_logits_pr.append(np.concatenate(sub_logits))
  comp_logits_pr.append(np.concatenate(comp_logits))
  sent_logits_pr.append(np.concatenate(sent_logits))

top_thresholds = [0.1969960277715784, 0.22856181358436206, 0.18800807278012546, 0.42076735866421744, 0.188429153663433, 0.24591746689126756, 0.20983471186315358, 0.4365670657497739, 0.28280920588737424]
sub_thresholds = [0.21957855014040917, 0.19366260893855475, 0.3255171298899038, 0.33183340156273944, 0.4857037846005992, 0.15174583400932873, 0.46007111169795345, 0.2251651060265043, 0.19376815494901492, 0.37150065067605836, 0.18189744496786836, 0.24164270074532695, 0.19298219650317167, 0.37682619636156556, 0.33005044535839734, 0.21131058711565034, 0.36170617432524893, 0.15047266734171708, 0.22541301215386067, 0.1947837737010123, 0.4699332847555047, 0.310423688681474, 0.36948024421953063, 0.16785958723525482, 0.39575850055892337, 0.24538317976204244, 0.4431052867303466, 0.19942697545930543, 0.47837218061864795]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sentiment_from_logits(sent_logits, T=1.2):
    delta = sent_logits[..., 1] - sent_logits[..., 0]
    sent = np.ones(delta.shape, dtype=np.int64)
    sent[delta < 0] = -1
    conf = sigmoid(np.abs(delta))/T
    return sent, conf

def inference(top_to_sub_dense, top_logits, sub_logits, comp_logits, sent_logits, top_thresholds,
              sub_thresholds, comp_thresholds=0.7):
    p_top = sigmoid(top_logits)        # (B, T)
    p_sub = sigmoid(sub_logits)        # (B, S)
    p_comp = sigmoid(comp_logits)      # (B, C)
    top_thr = np.asarray(top_thresholds)   # (1, T)
    sub_thr = np.asarray(sub_thresholds)   # (1, S)
    pred_top_bin = (p_top > top_thr).astype(np.float32)
    mask = np.matmul(pred_top_bin, top_to_sub_dense)  # (B, S)
    mask = mask.clip(0.0, 1.0)
    no_top = np.sum(pred_top_bin, axis=1, keepdims=True) == 0
    mask = mask + no_top * (1.0 - mask)
    p_sub_masked = p_sub * mask
    pred_sub_bin = (p_sub_masked > sub_thr).astype(np.float32)
    pred_comp_bin = ((p_comp > comp_thresholds)*pred_sub_bin).astype(np.float32)
    sent_preds, sent_conf = sentiment_from_logits(sent_logits)

    return pred_top_bin, pred_sub_bin, pred_comp_bin, sent_preds, sent_conf

CFG = {
    "hierarchical_json": r"/content/final_aspa_data_hierarchical_with_sentiments_temp_v6.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
DEVICE = "cuda"
top_to_sub_dense = np.load("/content/top_to_sub_dense (1).npy")

pred_top_bin, pred_sub_bin, pred_comp_bin, sent_preds, sent_conf = [], [], [], [], []
for i in range(len(top_logits_pr)):
   outputs = inference(top_to_sub_dense, top_logits_pr[i], sub_logits_pr[i], comp_logits_pr[i], sent_logits_pr[i], top_thresholds,
              sub_thresholds, comp_thresholds=0.7)
   pred_top_bin.append(outputs[0])
   pred_sub_bin.append(outputs[1])
   pred_comp_bin.append(outputs[2])
   sent_preds.append(outputs[3])
   sent_conf.append(outputs[4])

def robust_sign(x: np.ndarray, eps: float = 1e-6):
    s = np.zeros_like(x)
    s[x > eps] = 1
    s[x < -eps] = -1
    return s

def weighted_sentiment_score(
    sentiments: np.ndarray,
    aspect_mask: np.ndarray,
    weights: np.ndarray,
    aspect_reliability: float,
    eps: float = 1e-8) -> float:
    """
    S_j: weighted belief score
    """
    if aspect_mask.sum() == 0:
        return 0.0

    num = np.sum(sentiments * aspect_mask * weights, axis=0)
    den = np.sum(weights * aspect_mask, axis=0) + eps
    return aspect_reliability * (num / den)


def g_conf(conf: np.ndarray, alpha: float):
    # conf = |logit_pos - logit_neg|
    return 1.0 - np.exp(-conf / alpha)

def h_useful(u: np.ndarray, beta: float = 1.0):
    return np.log1p(beta * u)

def build_weights(
    sent_conf: np.ndarray,   # (N, A)
    usefulness: np.ndarray,  # (N,)
    alpha: float,
    beta: float):
    return g_conf(sent_conf, alpha) * h_useful(usefulness[:, None], beta)

def build_weights_norm(
    sent_conf: np.ndarray,   # (N, A)
    usefulness: np.ndarray,  # (N,)
    alpha: float,
    beta: float):
    return g_conf(sent_conf/(np.median(sent_conf)+1e-8), alpha) * h_useful(usefulness[:, None]/(np.median(usefulness[:, None])+1e-8), beta)

def final_confidence(
    usefulness: np.ndarray, #(n, 1)
    aspect_mask: np.ndarray, #(n, 29)
    sentiments: np.ndarray, #(n, 29)
    sent_preds, #(n, 29)
    weights: np.ndarray, #(n, 29)
    aspect_reliability: float,
    beta: float = 1.0,
    gamma: float = 1.0):
    """
    Final confidence for aspect j
    """
    weighted_sentiments = sentiments * aspect_mask * weights
    support = np.sum(weighted_sentiments, axis=0)
    total_weight = np.sum(np.abs(weighted_sentiments), axis=0) + 1e-8
    agreement = abs(support) / total_weight
    usefulness = usefulness[:, None]
    effective_volume = np.sum(aspect_mask * h_useful(usefulness, beta), axis=0)

    volume_term = 1.0 - np.exp(-effective_volume / gamma)

    disagree = np.zeros_like(sentiments) + 1e-8
    disagree[sent_preds != robust_sign(sentiments)] = 1.0
    disp_num = np.sum(aspect_mask * usefulness * disagree, axis=0)
    disp_den = np.sum(aspect_mask * usefulness, axis=0) + 1e-8
    eps = 0.65
    kappa = 0.8
    sentiment_reliability = eps + (1 - eps) * (aspect_reliability ** kappa)


    dispersion = disp_num / disp_den
    confidence = (
        sentiment_reliability *
        agreement *
        volume_term *
        (1.0 - dispersion)
    )

    return confidence

sent_preds = sent_preds[0]
pred_sub_bin = pred_sub_bin[0]
sent_conf = sent_conf[0]
usefulness = rank_final[0][rank_final[0]>5]

alpha, beta, gamma, phi = 0.8, 0.6, 24, 0.2

weights_norm = build_weights_norm(sent_conf, usefulness, alpha, beta)
weights = build_weights(sent_conf, usefulness, alpha, beta)
sent = weighted_sentiment_score(sent_preds, pred_sub_bin, weights_norm, aspect_reliability)
sentiments = robust_sign(sent) # here sentiments are the final sentiments predicted using weighted technique
conf= final_confidence(usefulness, pred_sub_bin, sent_preds, sentiments, weights, # whereas sent_preds are the prediction of ealier raw sentiments
      aspect_reliability, beta = beta, gamma = gamma)
confidence = 100 * (conf ** phi)

top_labels={0: 'battery & charging',
 1: 'build quality & design',
 2: 'camera',
 3: 'customer experience',
 4: 'device features & functionality',
 5: 'device performance',
 6: 'display',
 7: 'price & value',
 8: 'software & ux'}

sub_labels = {0: 'ai assistants & smart features',
 1: 'audio & speakers',
 2: 'battery & charging comparisons',
 3: 'battery capacity',
 4: 'battery charging speed',
 5: 'battery life & health',
 6: 'build quality, design, durability & ergonomics',
 7: 'camera features',
 8: 'camera quality',
 9: 'connectivity & wireless standards',
 10: 'customer service & warranty',
 11: 'customization & ui',
 12: 'display aesthetics',
 13: 'display defects',
 14: 'display quality',
 15: 'display visual performance',
 16: 'essentials & expansion',
 17: 'general features',
 18: 'processor performance',
 19: 'product perception',
 20: 'repair & replacement costs',
 21: 'software experience, features & system settings',
 22: 'software issues & bugs',
 23: 'software updates & support longevity',
 24: 'stylus & input accessories',
 25: 'technical performance & storage specs',
 26: 'thermals',
 27: 'user experience',
 28: 'value for money & deals'}

def get_best_reviews(i, product_data):
  likes, best_reviews = [], []
  indx_sub_asp_act = pred_sub_bin[:, i] == 1.0
  bool_indx = indx_sub_asp_act & (usefulness > 8.75)
  indx = np.where(bool_indx == True)[0]
  l = len(indx)
  if l > 3:
    for j in indx:
      text_dict = product_data[0][j]
      best_reviews.append(text_dict.get("text"))
    return best_reviews
  elif l == 0: return []
  else:
    for j in indx:
      best_reviews.append(product_data[0][j].get("text"))
    return best_reviews

def get_specs(i, product_specs): # here i is top aspects indx connext connect top aspects and ts sub aspects to the specs
  specs = {}
  labels = top_to_specs[int(i)]
  specs_dict = product_specs["specifications"]
  for specs_label in labels:
    if specs_label in specs_dict:
      specs[specs_label] = specs_dict[specs_label]
  return specs

input_llm = {"product_info": {"product_name": product_name,
                              "product_type": "Latest Smartphones"}}
input_llm["product_summary"] = [{"top_aspect_name": top_labels[nm], "sub_aspects": [],
                                 "specifications": get_specs(i, data_specs)} for i, nm in enumerate(top_labels)]

for i, sub_asp_conf in enumerate(confidence):
  useful_sentences = []
  sub_aspect_dict = {}
  if sub_asp_conf != 0.0:
    sub_aspect_dict["sub_aspect_name"] = sub_labels[i]
    sub_aspect_dict["sub_aspect_sentiment"] = "positive" if sentiments[i] > 0 else "negative"
    sub_aspect_dict["sub_aspect_confidence"] = int(sub_asp_conf+1.0)
    sub_aspect_dict["sub_aspect_top_reviews"] = get_best_reviews(i, product_data)
    active_top = np.where(top_to_sub_dense[:, i] == 1)[0][0]
    input_llm["product_summary"][active_top]["sub_aspects"].append(sub_aspect_dict)

input_llm