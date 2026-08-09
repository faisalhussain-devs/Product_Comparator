import numpy as np
from pathlib import Path

TOP_LABELS= {0: 'battery & charging',
 1: 'build quality & design',
 2: 'camera',
 3: 'customer experience',
 4: 'device features & functionality',
 5: 'device performance',
 6: 'display',
 7: 'price & value',
 8: 'software & ux'}

SUB_LABELS = {0: 'ai assistants & smart features',
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

ASPECT_RELIABILITY = np.array([
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
], dtype=np.float32)

TOP_TO_SPECS = {
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

TOP_THRESHOLDS = np.array([0.1969960277715784, 0.22856181358436206, 0.18800807278012546, 0.42076735866421744, 0.188429153663433, 0.24591746689126756, 0.20983471186315358, 0.4365670657497739, 0.28280920588737424], dtype=np.float32)
SUB_THRESHOLDS = np.array([0.21957855014040917, 0.19366260893855475, 0.3255171298899038, 0.33183340156273944, 0.4857037846005992, 0.15174583400932873, 0.46007111169795345, 0.2251651060265043, 0.19376815494901492, 0.37150065067605836, 0.18189744496786836, 0.24164270074532695, 0.19298219650317167, 0.37682619636156556, 0.33005044535839734, 0.21131058711565034, 0.36170617432524893, 0.15047266734171708, 0.22541301215386067, 0.1947837737010123, 0.4699332847555047, 0.310423688681474, 0.36948024421953063, 0.16785958723525482, 0.39575850055892337, 0.24538317976204244, 0.4431052867303466, 0.19942697545930543, 0.47837218061864795], dtype=np.float32)

HF_REPO_ID = "Faisal191/aspect-classifier"

RANKER_TOKENIZER_SUBFOLDER = "RankingReviews"
RANKER_MODEL_FILE = "RankingReviews/model.onnx"

ABSA_TOKENIZER_SUBFOLDER = "Domain_trained_encoder"
ABSA_MODEL_FILE = "HABSA/Habsa_v6_fp32.onnx"

RANK_BATCH_SIZE = 128
ABSA_BATCH_SIZE = 128

MAX_RANK_SEQUENCE_LENGTH = 100
MAX_ABSA_SEQUENCE_LENGTH = 256

COMPARISON_THRESHOLD = 0.7

CONFIDENCE_ALPHA = 0.8
CONFIDENCE_BETA = 0.6
CONFIDENCE_GAMMA = 24.0
CONFIDENCE_PHI = 0.2

SENTIMENT_TEMPERATURE = 1.2
REVIEW_USEFULNESS_THRESHOLD = 8.75

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOP_TO_SUB_PATH = PROJECT_ROOT / "assets" / "top_to_sub_dense (1).npy"

if len(TOP_LABELS) != 9:
    raise ValueError("Expected 9 top-level aspects.")

if len(SUB_LABELS) != 29:
    raise ValueError("Expected 29 sub-aspects.")

if len(TOP_THRESHOLDS) != len(TOP_LABELS):
    raise ValueError("Top threshold count does not match top aspect count.")

if len(SUB_THRESHOLDS) != len(SUB_LABELS):
    raise ValueError("Sub threshold count does not match sub-aspect count.")

if len(ASPECT_RELIABILITY) != len(SUB_LABELS):
    raise ValueError("Aspect reliability count does not match sub-aspect count.")

if set(TOP_TO_SPECS) != set(TOP_LABELS):
    raise ValueError("TOP_TO_SPECS keys do not match TOP_LABELS.")

if not np.all((TOP_THRESHOLDS >= 0) & (TOP_THRESHOLDS <= 1)):
    raise ValueError("Invalid top-level threshold.")

if not np.all((SUB_THRESHOLDS >= 0) & (SUB_THRESHOLDS <= 1)):
    raise ValueError("Invalid sub-aspect threshold.")

if not np.all((ASPECT_RELIABILITY >= 0) & (ASPECT_RELIABILITY <= 1)):
    raise ValueError("Invalid aspect reliability value.")

if not TOP_TO_SUB_PATH.exists():
    raise FileNotFoundError(
        f"Missing top-to-sub mapping: {TOP_TO_SUB_PATH}"
    )
top_to_sub = np.load(TOP_TO_SUB_PATH)

if top_to_sub.shape != (9, 29):
    raise ValueError(
        f"Expected top-to-sub matrix shape (9, 29), "
        f"got {top_to_sub.shape}."
    )