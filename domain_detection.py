import re

def looks_like_planning(text: str) -> bool:
    t = text.lower()

    strong_markers = [
        "[statement]",
        "[plan]",
        "as initial conditions i have that",
        "my goal is to have that",
        "my plan is as follows",
        "here are the actions i can do",
        "here are the actions that can be performed",
        "i have the following restrictions on my actions",
        "the following are the restrictions on the actions",
    ]

    action_markers = [
        "pick up", "unstack", "put down", "stack",
        "load", "unload", "drive", "fly",
        "lift", "drop",
        "attack", "feast", "succumb", "overcome",
        "paltry", "sip", "clip", "wretched", "memory", "tightfisted",
    ]

    strong_hits = sum(m in t for m in strong_markers)
    action_hits = sum(m in t for m in action_markers)

    if strong_hits >= 2:
        return True
    if strong_hits >= 1 and action_hits >= 2:
        return True

    return False
    
def _contains_any(text, phrases):
    return any(p in text for p in phrases)

def _count_hits(text, phrases):
    return sum(1 for p in phrases if p in text)

def detect_planning_subtype(text: str) -> str:
    t = text.lower().strip()

    # --------------------------------------------------
    # 1) Blocks world
    # --------------------------------------------------
    blocks_markers = [
        "pick up a block",
        "unstack a block",
        "put down a block",
        "stack a block",
        "the hand is empty",
        "on top of",
        "on the table",
        "the block is clear",
        "clear",
        "block"
    ]
    blocks_hits = _count_hits(t, blocks_markers)

    if blocks_hits >= 4 and _contains_any(
        t, ["pick up a block", "unstack a block", "stack a block", "put down a block"]
    ):
        return "blocks_world"

    # --------------------------------------------------
    # 2) Depot logistics with hoists / crates / pallets
    # --------------------------------------------------
    depot_markers = [
        "depot", "distributor", "hoist", "crate", "pallet", "truck",
        "lift a crate", "drop a crate", "load a crate", "unload a crate",
        "drive a truck", "surface", "available", "clear"
    ]
    depot_hits = _count_hits(t, depot_markers)

    if depot_hits >= 5 and _contains_any(t, ["hoist", "crate", "pallet", "depot"]):
        return "depot_logistics"

    # --------------------------------------------------
    # 3) Air cargo / transport with trucks + airplanes + cities
    # --------------------------------------------------
    air_markers = [
        "package", "truck", "airplane", "airport", "city", "cities",
        "load a package into a truck", "load a package into an airplane",
        "unload a package from a truck", "unload a package from an airplane",
        "drive a truck", "fly an airplane", "location_", "city_"
    ]
    air_hits = _count_hits(t, air_markers)

    if air_hits >= 5 and _contains_any(t, ["airplane", "airport", "fly an airplane"]):
        return "air_cargo_logistics"

    # --------------------------------------------------
    # 4) Abstract operator planning
    # Catch synthetic verb domains and explicit operator schemas
    # --------------------------------------------------
    abstract_markers = [
        "here are the actions i can do",
        "the following restrictions on my actions",
        "once",
        "to perform",
        "the following facts need to be true",
        "the following facts will be true",
        "the following facts will be false",
        "[statement]",
        "[plan]"
    ]
    abstract_hits = _count_hits(t, abstract_markers)

    nonsense_action_markers = [
        "attack object", "feast object", "succumb object", "overcome object",
        "paltry", "sip", "clip", "wretched", "memory", "tightfisted"
    ]
    nonsense_hits = _count_hits(t, nonsense_action_markers)

    if nonsense_hits >= 1:
        return "abstract_operator_planning"

    if abstract_hits >= 5:
        return "abstract_operator_planning"

    # --------------------------------------------------
    # Fallbacks
    # --------------------------------------------------
    if _contains_any(t, ["block", "on top of", "on the table", "hand is empty"]):
        return "blocks_world"

    if _contains_any(t, ["hoist", "crate", "pallet", "depot", "distributor"]):
        return "depot_logistics"

    if _contains_any(t, ["airplane", "airport", "package", "city_", "fly an airplane"]):
        return "air_cargo_logistics"

    return "abstract_operator_planning"
    
def looks_like_future_prediction(text: str) -> bool:
    t = text.lower()

    strong_markers = [
        "you are an agent that can predict future events",
        "the event to be predicted:",
        "\\boxed{",
        "do not refuse to make a prediction",
        'do not say "i cannot predict the future."',
        "you must make a clear prediction based on the best data currently available",
    ]

    medium_markers = [
        "resolved around",
        "important: your final answer must end with this exact format:",
        "\\boxed{yes} or \\boxed{no}",
        "listing all plausible options you have identified",
        "or \\boxed{",
    ]

    strong_hits = sum(m in t for m in strong_markers)
    medium_hits = sum(m in t for m in medium_markers)

    if strong_hits >= 2:
        return True
    if strong_hits >= 1 and medium_hits >= 2:
        return True

    return False
    
def _contains_any(text, phrases):
    return any(p in text for p in phrases)

def _count_hits(text, phrases):
    return sum(1 for p in phrases if p in text)

def detect_future_prediction_subtype(text: str) -> str:
    t = text.lower().strip()

    # --------------------------------------------------
    # 1) Binary yes/no forecasts
    # --------------------------------------------------
    yes_no_markers = [
        r"\boxed{yes} or \boxed{no}",
        "will ",
        "soft touchdown",
        "reach",
        "be net positive",
        "win the green jersey",
        "get 2 yellow cards",
        "approval be net positive"
    ]
    if _contains_any(t, [r"\boxed{yes} or \boxed{no}"]):
        return "binary_outcome_forecast"

    # --------------------------------------------------
    # 2) Multiple-choice boxed forecasts
    # --------------------------------------------------
    mc_markers = [
        "a.  the outcome be",
        "b.  the outcome be",
        "c.  the outcome be",
        "listing all plausible options you have identified",
        r"\boxed{a}",
        r"\boxed{b, c}"
    ]
    if _contains_any(t, mc_markers):
        # Sports head-to-head with labeled outcomes
        if _contains_any(t, [" vs. ", "tour de france", "eisners", "cpi in july", "tweet likes"]):
            if " vs. " in t:
                return "sports_match_forecast"
            return "multiple_choice_forecast"

    # --------------------------------------------------
    # 3) Sports match / rider / race comparisons
    # --------------------------------------------------
    sports_markers = [
        " vs. ",
        "tour de france",
        "rider",
        "will place better",
        "green jersey",
        "feyenoord", "fenerbahçe", "amazonas", "goiás",
        "corinthians", "fortaleza", "cfr cluj", "sporting braga",
        "volta redonda", "novorizontino"
    ]
    sports_hits = _count_hits(t, sports_markers)
    if sports_hits >= 2:
        return "sports_match_forecast"

    # --------------------------------------------------
    # 4) Ranked list / leaderboard / top-k forecasts
    # --------------------------------------------------
    ranked_markers = [
        "前3名", "前五", "前十名", "top 3", "top 5", "top 10",
        "排名前五", "排名前三", "日榜", "排行榜", "榜第一名",
        "只回答车型", "只回答视频号名称", "只回答作者名称",
        "只回答歌曲名", "用英文回答"
    ]
    ranked_hits = _count_hits(t, ranked_markers)

    media_markers = [
        "qq音乐", "酷狗音乐", "hulu", "猫眼电影", "新榜", "微博指数",
        "视频号指数", "小红书指数", "快手指数", "公众号", "github开源项目日飙升榜",
        "steam公布的support stats", "懂车帝", "kolrank", "box office mojo"
    ]
    media_hits = _count_hits(t, media_markers)

    if ranked_hits >= 2 and media_hits >= 1:
        return "media_chart_forecast"

    if ranked_hits >= 2:
        return "ranked_list_forecast"

    # --------------------------------------------------
    # 5) Numeric market forecasts
    # --------------------------------------------------
    market_markers = [
        "收盘价", "开盘价", "当日最高点", "股票", "总市值", "市价总值",
        "指数", "hang seng", "aapl", "sz：", "sh：", "沪深300", "日经平均股价指数",
        "深圳证券交易所", "上海证券交易所", "港股通", "coinmarketcap fear and greed index"
    ]
    market_hits = _count_hits(t, market_markers)

    if market_hits >= 2:
        return "numeric_market_forecast"

    # --------------------------------------------------
    # 6) Numeric metric forecasts
    # --------------------------------------------------
    metric_markers = [
        "价格是多少", "平均价格", "指数是多少", "是多少元/公斤", "第一个数字是多少",
        "占有率是%多少", "热度是多少万", "gross是多少美元", "number of riders finishing",
        "农产品批发价格", "support stats", "fdi综合指数", "煤炭运价指数", "搜索引擎排行榜",
        "操作系统排行榜", "box office", "daily box office"
    ]
    metric_hits = _count_hits(t, metric_markers)

    if metric_hits >= 2:
        return "numeric_metric_forecast"

    # --------------------------------------------------
    # 7) Fallbacks
    # --------------------------------------------------
    if _contains_any(t, ["前3名", "前五", "前十名", "榜第一名", "排行榜", "日榜"]):
        return "ranked_list_forecast"

    if _contains_any(t, ["收盘价", "开盘价", "总市值", "最高点", "指数是多少"]):
        return "numeric_market_forecast"

    if _contains_any(t, ["价格是多少", "平均价格", "第一个数字是多少", "占有率是%多少"]):
        return "numeric_metric_forecast"

    return "multiple_choice_forecast"
    
def looks_like_coding(text: str) -> bool:
    t = text.lower()

    strong_markers = [
        "you should write self-contained code starting with:",
        "def task_func(",
        "the function should output with:",
        "```",
    ]

    medium_markers = [
        "the function should raise the exception for:",
        "import pandas",
        "import numpy",
        "import matplotlib",
        "import seaborn",
        "import requests",
        "import sqlite3",
        "import os",
        "import re",
        "import json",
        "from sklearn",
        "from datetime import",
        "subprocess",
        "dataframe",
        "axes",
        "figure",
    ]

    strong_hits = sum(m in t for m in strong_markers)
    medium_hits = sum(m in t for m in medium_markers)

    if strong_hits >= 2:
        return True
    if strong_hits >= 1 and medium_hits >= 2:
        return True

    return False
    
def _contains_any(text, phrases):
    return any(p in text for p in phrases)

def _count_hits(text, phrases):
    return sum(1 for p in phrases if p in text)

def detect_coding_subtype(text: str) -> str:
    t = text.lower().strip()

    # --------------------------------------------------
    # 1) Web / API / scraping / networked app behavior
    # --------------------------------------------------
    web_markers = [
        "api", "url", "http", "https", "requests", "urllib", "fetch",
        "html", "anchor tags", "scrape", "scrapes", "web page", "webpage",
        "beautifulsoup", "pyquery", "github api", "downloads the file",
        "smtp", "flask", "send emails", "mail", "geolocation", "ip api"
    ]
    web_hits = _count_hits(t, web_markers)

    if web_hits >= 2:
        return "web_api_scraping"

    # --------------------------------------------------
    # 2) ML / stats / forecasting
    # --------------------------------------------------
    ml_markers = [
        "linear regression", "kmeans", "pca", "principal component",
        "arima", "forecast", "forecasting", "chi-square", "chi square",
        "contingency table", "standardscaler", "scale the", "scaled",
        "cluster", "clustering", "fitted linear model", "principal components",
        "independence", "covariance matrix"
    ]
    ml_hits = _count_hits(t, ml_markers)

    if ml_hits >= 1:
        return "ml_stats_forecasting"

    # --------------------------------------------------
    # 3) Text / JSON / regex processing
    # --------------------------------------------------
    text_markers = [
        "json string", "json-formatted", "json", "regex", "regular expression",
        "remove punctuation", "lowercase", "word frequency", "split into words",
        "scramble the letters", "special characters", "normalize whitespace",
        "extract all urls", "extract recepient email", "count the occurrence",
        "counts the number of words", "formatted date and time string",
        "word counts", "convert to lowercase", "randomizing character casing",
        "string representations", "parses a json string"
    ]
    text_hits = _count_hits(t, text_markers)

    if text_hits >= 2:
        return "text_json_regex_processing"

    # --------------------------------------------------
    # 4) DataFrame / tabular processing
    # --------------------------------------------------
    df_markers = [
        "dataframe", "pandas", "pd.", "csv", "sqlite", "sql query",
        "table", "columns", "grouped by", "groupby", "crosstab",
        "pivot", "read_csv", "read_sql_query", "export to a csv",
        "query an sqlite database", "containing categorical data",
        "column names", "tabular", "generate a pandas dataframe",
        "process a pandas dataframe", "read sqlite3 table via pandas"
    ]
    df_hits = _count_hits(t, df_markers)

    if df_hits >= 2:
        return "dataframe_tabular_processing"

    # --------------------------------------------------
    # 5) Visualization / plotting
    # Only when plotting is the main ask
    # --------------------------------------------------
    viz_markers = [
        "plot", "plots", "chart", "graph", "heatmap", "histogram",
        "scatter plot", "pairplot", "pair plot", "pie chart",
        "bar chart", "line plot", "subplot", "axes", "figure",
        "matplotlib", "seaborn", "draw", "visualize", "visualization"
    ]
    viz_hits = _count_hits(t, viz_markers)

    if viz_hits >= 2:
        return "visualization_plotting"

    # --------------------------------------------------
    # 6) Filesystem / OS operations
    # Lower priority because many tasks touch files incidentally
    # --------------------------------------------------
    fs_markers = [
        "directory", "directories", "file", "files", "folder", "path",
        "os.walk", "glob", "shutil", "subprocess", "tar", "archive",
        ".bat", ".json", "move", "moves", "rename", "compress", "gzip",
        "root_dir", "dest_dir", "source_directory", "destination_directory",
        "md5 hash", "hash value", "rglob", "pathlib", "permissionerror"
    ]
    fs_hits = _count_hits(t, fs_markers)

    if fs_hits >= 3 and _contains_any(
        t, ["directory", "file", "files", "folder", "archive", "move", "compress"]
    ):
        return "filesystem_os_ops"

    # --------------------------------------------------
    # 7) Fallback utility
    # --------------------------------------------------
    return "algorithmic_utility"
    
NUMBER_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","billion",
    "half","double","triple","twice","thrice"
}

def looks_like_math(text: str) -> bool:
    raw = text
    t = text.lower()

    # --------------------------------------------------
    # Hard exclusions — these belong to other domains
    # --------------------------------------------------
    anti_markers = [
        "you should write self-contained code starting with:",
        "def task_func(",
        "the function should output with:",
        "you are an agent that can predict future events",
        "the event to be predicted:",
        "do not refuse to make a prediction",
        "[statement]",
        "[plan]",
        "as initial conditions i have that",
        "my goal is to have that",
        "my plan is as follows",
        "here are the actions i can do",
        "here are the actions that can be performed",
        "the following are the restrictions on the actions",
        "i have the following restrictions on my actions",
    ]
    if any(m in t for m in anti_markers):
        return False

    # --------------------------------------------------
    # Symbolic / formal math cues
    # --------------------------------------------------
    symbolic_markers = [
        "$", "\\[", "\\]", "\\frac", "\\sqrt", "\\triangle", "\\angle",
        "\\sin", "\\cos", "\\tan", "\\log", "\\binom", "\\overline",
        "[asy]", ".png", "coordinate", "vertices", "equation", "roots",
        "relatively prime", "in lowest terms", "divisible by", "remainder when",
        "greatest common divisor"
    ]

    # --------------------------------------------------
    # Explicit math-topic cues
    # --------------------------------------------------
    topic_markers = [
        "integer", "integers", "positive integer", "prime", "factor",
        "probability", "ratio", "percent", "percentage",
        "sum", "product", "difference", "remainder",
        "sequence", "series", "average", "mean",
        "area", "perimeter", "radius", "diameter",
        "triangle", "quadrilateral", "circle", "sphere", "polygon",
        "committee", "distinct", "random", "equally likely",
        "divisors", "multiple", "least", "greatest", "smallest", "largest",
    ]

    # --------------------------------------------------
    # Arithmetic / word-problem operation cues
    # --------------------------------------------------
    operation_markers = [
        "each", "every", "altogether", "total", "remaining", "left",
        "shared equally", "equally", "split", "divide", "divided",
        "times as many", "more than", "less than", "fewer than",
        "per hour", "per day", "per week", "per month", "per year",
        "costs", "price", "earn", "earned", "spend", "spent", "budget",
        "hour", "hours", "minute", "minutes", "day", "days", "week", "weeks",
        "year", "years", "mile", "miles", "foot", "feet",
        "pound", "pounds", "kilogram", "kilograms",
        "cup", "cups", "slice", "slices", "page", "pages",
        "dollar", "dollars", "$", "%",
        # added — catches GSM8K problems missed before
        "apples", "oranges", "candies", "cookies", "bags", "boxes",
        "bottles", "cans", "eggs", "birds", "cats", "dogs", "fish",
        "marbles", "stickers", "toys", "trees", "flowers",
        "kilograms", "liters", "centimeters", "meters",
        "per day", "per hour", "per week",
        "faster", "slower", "together", "combined",
        "win ratio", "win rate", "shots", "rate",
        "paint", "painted", "fence", "monkeys", "pile",
    ]

    # --------------------------------------------------
    # Question-style cues
    # --------------------------------------------------
    question_markers = [
        "how many", "how much", "find", "compute", "determine", "what is the",
        "how far", "how long", "how fast", "how old",
        "what is", "calculate", "evaluate", "simplify", "solve",
    ]

    symbolic_hits = sum(m in raw for m in symbolic_markers)
    topic_hits    = sum(m in t for m in topic_markers)
    op_hits       = sum(m in t for m in operation_markers)
    q_hits        = sum(m in t for m in question_markers)

    has_digit_number = bool(re.search(r"\b\d+(\.\d+)?\b", raw))
    has_number_word  = any(re.search(rf"\b{re.escape(w)}\b", t) for w in NUMBER_WORDS)
    has_number_like  = has_digit_number or has_number_word
    has_equals       = "=" in raw
    has_coords       = bool(re.search(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", raw))

    # --------------------------------------------------
    # Decision rules — ordered strongest to weakest
    # --------------------------------------------------

    # Strong symbolic / competition math
    if symbolic_hits >= 2:
        return True
    if symbolic_hits >= 1 and (topic_hits >= 1 or has_equals or has_coords):
        return True

    # Plain-English arithmetic / counting / rate / ratio
    if has_number_like and q_hits >= 1 and (topic_hits >= 1 or op_hits >= 2):
        return True

    # Short number-theory / combinatorics style
    if has_number_like and topic_hits >= 2:
        return True

    # GSM8K style — numbers + operation context alone is enough
    if has_number_like and op_hits >= 3:
        return True

    return False

def _contains_any(text, phrases):
    return any(p in text for p in phrases)

def _count_hits(text, phrases):
    return sum(1 for p in phrases if p in text)

def detect_math_subtype(text: str) -> str:
    t = text.lower().strip()

    # --------------------------------------------------
    # has_latex: True only for real math LaTeX.
    # Bare dollar amounts like "$100", "$1750" must NOT qualify —
    # only math variables ($x, $n) and backslash commands count.
    # --------------------------------------------------
    has_latex_command = bool(re.search(r'\\\w+|\\\[|\\\]|\\begin', text))
    has_math_dollar   = bool(re.search(r'\$[^0-9\s$.]', text))  # $x, $n — not $100
    has_latex = has_latex_command or has_math_dollar

    # --------------------------------------------------
    # 1) Geometry
    # --------------------------------------------------
    geometry_markers = [
        "triangle", "\\triangle", "quadrilateral", "pentagon", "hexagon",
        "circle", "chord", "tangent", "radius", "diameter", "perimeter",
        "area", "convex", "polygon", "median", "bisector", "angle",
        "parallel", "rectangle", "square", "parallelogram",
        "sphere", "cone", "pyramid", "surface", "volume",
        "inscribed", "circumscribed", "point p", "vertices", "centers",
        "flat surface", "inside triangle", "on side", "collinear",
        "rotation", "reflection", "altitude", "equilateral",
        "line with slope", "perpendicular", "coordinate system",
    ]
    geometry_hits = _count_hits(t, geometry_markers)
    if geometry_hits >= 2:
        return "geometry"

    # --------------------------------------------------
    # 2) Sequence / recursive / functional
    # Ambiguous token markers (a_n, f(x), etc.) are gated
    # behind has_latex to prevent substring matches inside
    # plain-English words ("can", "than", "plan" contain "a_n").
    # k-th variants cover the LaTeX spaced form "$k$ -th".
    # --------------------------------------------------
    seq_markers_always = [
        "sequence", "define a sequence", "recursively",
        "x_{n+1}", "\\begin{cases}", "piecewise",
        "define a sequence as follows",
        "k-th centimeter", "k-th picket", "k-th term",
        "k$ -th", "k$-th",                              # LaTeX: "$k$ -th centimeter"
        "a_1 = 2", "a_1 = 5", "a_2 = 2", "a_2 = 5",   # explicit indexed-term assignments
    ]
    seq_markers_latex_only = [
        # Valid seq tokens but unsafe as plain-text substrings
        "x_1", "x_{n+1}", "a_0", "a_1", "a_2", "a_n", "b_n", "s_n",
        "let s_n", "let d(x)", "f(x)",
        "smallest n such that", "indexed so that",
        "for all positive integers",
    ]

    seq_hits = _count_hits(t, seq_markers_always)
    if has_latex:
        seq_hits += _count_hits(t, seq_markers_latex_only)

    seq_confirm = _contains_any(t, [
        "sequence", "recursively", "x_{n+1}", "\\begin{cases}",
        "define a sequence as follows", "k-th", "k$ -th", "k$-th",
    ])
    if has_latex:
        seq_confirm = seq_confirm or _contains_any(t, [
            "a_n", "f(x)", "a_0", "a_1", "a_2", "indexed so that",
        ])

    if seq_hits >= 1 and seq_confirm:
        return "sequence_recursive_functional"

    # --------------------------------------------------
    # 3) Counting / probability
    # Strong single signals route immediately before the
    # scored cp_markers list, catching dice/coin problems
    # that would otherwise leak into number theory via
    # "multiple of" or "remainder".
    # --------------------------------------------------
    if _contains_any(t, [
        "probability", "randomly", "equally likely",
        "fair coin", "fair die", "fair dice",
    ]):
        return "counting_probability"

    cp_markers = [
        "random", "arrangements", "ways", "sets of",
        "committee", "choose", "chosen", "subsets", "ordered triples",
        "ordered pairs", "inequivalent", "tilings",
        "independent", "tournament", "distinct convex polygons",
        "non-empty disjoint subsets", "permutation", "combination",
        "distinguishable", "indistinguishable", "sit around",
        "seat", "seating", "round table", "circular",
        "how many even", "how many integers", "how many positive",
        "how many ways", "how many different", "how many distinct",
        "in how many", "for how many",
        "binary integers", "binary strings",
        "number of possible sets", "number of possible",
        "even integers between", "integers between",
    ]
    cp_hits = _count_hits(t, cp_markers)
    if cp_hits >= 2:
        return "counting_probability"

    how_many = _contains_any(t, ["how many", "number of", "in how many", "for how many"])
    has_combo_context = _contains_any(t, [
        "ways", "arrangements", "subsets", "ordered", "committee",
        "choose", "sit", "seat", "round", "circular", "permut", "combin",
        "inequivalent", "tilings", "even integers", "integers between", "binary",
        "triples", "pairs", "values",
    ])
    if how_many and has_combo_context:
        return "counting_probability"

    if _contains_any(t, [
        "subsets", "ordered pairs", "ordered triples", "arrangements",
        "ways can", "committee", "tilings", "inequivalent",
        "sit around", "round table", "permutation", "combination",
    ]):
        return "counting_probability"

    # --------------------------------------------------
    # 4) Equation / expression manipulation
    # Runs BEFORE number theory so "relatively prime" in
    # competition answer format doesn't drag equation
    # problems into NT.
    # "is a factor of" replaces bare "factor of" to avoid
    # matching "prime factor of the integer".
    # --------------------------------------------------
    eq_markers = [
        "equation", "roots", "root", "real root", "real roots",
        "system of equations", "satisfy the equations", "polynomial",
        "is a factor of",
        "written in the form", "can be written in the form",
        "\\log", "log_", "\\sin", "\\cos", "\\tan", "complex",
        "product of the real roots", "suppose that x", "find x",
        "\\sqrt", "radical", "geometric progression",
    ]
    eq_hits = _count_hits(t, eq_markers)
    if eq_hits >= 1:
        return "equation_expression_manipulation"

    # --------------------------------------------------
    # 5) Number theory / digits / divisibility
    # "prime" excluded from nt_markers — too many equation
    # problems say "relatively prime". Handled via has_prime.
    # Weak path (nt_hits == 1) requires has_latex to block
    # plain GSM8K problems mentioning "digits" from landing here.
    # --------------------------------------------------
    nt_markers = [
        "divisible", "remainder", "mod", "modulo",
        "prime factor", "prime factors",
        "greatest common divisor", "gcd",
        "integer divisors", "divisors", "binary expansion", "leftmost digit",
        "digits are all different", "smallest positive integer",
        "non-zero digits", "sum of the digits",
        "positive odd integer divisors", "positive even integer divisors",
        "multiple of", "divided by",
        "number of digits", "how many digits",
        "\\pmod", "\\equiv",
    ]
    nt_hits = _count_hits(t, nt_markers)
    has_prime = "prime" in t and "relatively prime" not in t

    if nt_hits >= 2:
        return "number_theory_digit_divisibility"
    if nt_hits >= 1 and has_latex and _contains_any(t, [
        "digits", "divisible", "remainder", "\\pmod", "\\equiv", "multiple of",
    ]):
        return "number_theory_digit_divisibility"
    if has_prime and nt_hits >= 1 and has_latex:
        return "number_theory_digit_divisibility"

    # --------------------------------------------------
    # 6) Analytic / symbolic
    # --------------------------------------------------
    analytic_markers = [
        "graph of", "|x", "|y", "\\lfloor",
        "greatest integer that does not exceed",
        "absolute value", "floor", "region enclosed", "collinear again",
    ]
    analytic_hits = _count_hits(t, analytic_markers)
    if analytic_hits >= 1:
        return "analytic_symbolic_math"

    # --------------------------------------------------
    # 7) Arithmetic word problem
    # Scored marker list for common story-problem vocabulary.
    # --------------------------------------------------
    arithmetic_markers = [
        "dollars", "dollar", "cents", "budget", "spent", "cost",
        "earned", "earns", "hour", "hours", "minute", "minutes",
        "pages", "book", "week", "weeks", "day", "days", "month", "months",
        "year", "years",
        "sold", "buys", "bought", "left", "altogether", "total", "remaining",
        "allowance", "profit", "salary", "wage",
        "miles", "meters", "feet", "kilometers",
        "trees", "flowers", "people", "students",
        "money", "amount", "shared", "change",
        "pieces", "slices", "bike", "shoes", "clips", "stamps",
        "peanuts", "bananas", "reads", "rides", "meal", "post office",
        "apples", "oranges", "candies", "cookies", "cakes", "boxes",
        "bags", "bottles", "cans", "pounds", "kilograms", "liters",
        "marbles", "stickers", "cards", "toys", "eggs",
        "chickens", "fish", "cats", "dogs", "birds",
        "paint", "painted", "fence", "picket", "monkeys", "pile",
        "win ratio", "win rate", "matches", "shots", "speed", "rate",
        "faster", "slower", "together", "combined", "per day", "per hour",
        "letters", "hats", "gnomes", "houses", "average speed", "greatest average",
    ]
    arithmetic_hits = _count_hits(t, arithmetic_markers)

    if arithmetic_hits >= 3:
        return "arithmetic_word_problem"
    if not has_latex and arithmetic_hits >= 1:
        return "arithmetic_word_problem"

    # --------------------------------------------------
    # 8) Final safety net for no-LaTeX problems
    # Any plain-English problem with a question verb and a
    # number is almost certainly an arithmetic word problem —
    # there are no non-LaTeX competition problems in this dataset.
    # --------------------------------------------------
    has_question_verb = _contains_any(t, [
        "how many", "how much", "how far", "how long", "how fast",
        "calculate", "find the total", "what is the total",
        "how many more", "how many fewer", "how many less", "what is",
    ])
    if not has_latex and has_question_verb:
        return "arithmetic_word_problem"

    return "analytic_symbolic_math"


def phrase_hit(text: str, phrase: str) -> bool:
    return re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text) is not None

def count_phrase_hits(text: str, phrases) -> int:
    return sum(phrase_hit(text, p) for p in phrases)

NUMBER_WORDS = [
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","billion",
    "half","double","triple","twice"
]

def looks_like_common_sense(text: str) -> bool:
    raw = text
    t = " ".join(text.lower().split())

    # --------------------------------------------------
    # Hard exclusions for the three clean domains
    # --------------------------------------------------
    anti_markers = [
        "you should write self-contained code starting with:",
        "def task_func(",
        "the function should output with:",
        "you are an agent that can predict future events",
        "the event to be predicted:",
        "do not refuse to make a prediction",
        "[statement]",
        "[plan]",
        "as initial conditions i have that",
        "my plan is as follows",
        "here are the actions i can do",
        "here are the actions that can be performed",
    ]
    if any(m in t for m in anti_markers):
        return False

    # --------------------------------------------------
    # Explicit common-sense formats
    # --------------------------------------------------
    if "what is the best answer for the question among these?" in t:
        return True
    if "answer the question using the context" in t:
        return True

    # --------------------------------------------------
    # Strong symbolic math exclusions
    # --------------------------------------------------
    strong_math_markers = [
        "\\frac", "\\sqrt", "\\triangle", "\\angle", "\\sin", "\\cos", "\\tan",
        "\\log", "\\binom", "[asy]", "relatively prime", "in lowest terms",
        "remainder when", "divisible by", "positive integer",
        "prime factor", "real roots", "convex quadrilateral", "set of real numbers",
        "sequence as follows", "let $", "find the remainder", "find the product"
    ]
    if raw.count("$") >= 2:
        return False
    if any(m in raw.lower() for m in strong_math_markers):
        return False
 # --------------------------------------------------
    # Plain-English math question rejection
    # --------------------------------------------------
    math_question_markers = [
        "integer", "integers", "whole-number", "divisor", "divisors",
        "greatest common factor", "least common multiple",
        "perfect cube", "perfect square", "common fraction",
        "consecutive integers", "positive whole-number",
        "radius", "diameter", "sphere", "circle", "triangle",
        "quadrilateral", "polygon", "volume", "surface area",
        "area", "perimeter", "units"
    ]
    math_q_hits = count_phrase_hits(t, math_question_markers)

    # if it looks like a plain-English math question, reject common_sense
    if (t.startswith("how many ") or t.startswith("what is ") or t.startswith("what power ")) and math_q_hits >= 1:
        return False
    # --------------------------------------------------
    # Arithmetic-story math suppression
    # --------------------------------------------------
    arithmetic_markers = [
        "altogether", "remaining", "left", "total", "shared equally",
        "equally", "per hour", "per day", "per week", "per month", "per year",
        "times as many", "more than", "less than", "fewer than",
        "spent", "cost", "costs", "earned", "budget", "profit", "salary",
        "miles", "hours", "minutes", "days", "weeks", "months", "years",
        "cups", "pages", "pounds", "kilograms", "dollars", "cents",
        "average", "ratio", "percentage", "percent", "how many", "how much"
    ]
    arithmetic_hits = count_phrase_hits(t, arithmetic_markers)

    has_digit_number = bool(re.search(r"\b\d+(\.\d+)?\b", raw))
    has_number_word = any(phrase_hit(t, w) for w in NUMBER_WORDS)
    has_number_like = has_digit_number or has_number_word

    # --------------------------------------------------
    # Yes / no common-sense questions
    # --------------------------------------------------
    yesno_starts = [
        "is ", "are ", "does ", "did ", "can ", "could ", "would ",
        "was ", "were ", "has ", "have ", "do ", "will ", "if "
    ]
    starts_yesno = any(t.startswith(s) for s in yesno_starts)

    # --------------------------------------------------
    # Question cues
    # --------------------------------------------------
    wh_start = bool(re.match(
        r"^(which|who|what|where|when|why|how|in which|in what|approximately what percentage)\b", t
    ))

    question_cues = [
        " who ", " what ", " which ", " where ", " when ", " why ",
        " in which ", " in what ", " how old ", " how is ",
        " named after who", " based out of what", " based in which",
        " based in what", " located in which", " located in what",
        " starred in what", " directed by who", " played what",
        " better known", " what kind of ", " what type of ",
        " what city", " what country", " what county", " what state",
        " what year", " what network", " what age", " what profession",
        " what actor", " what actress", " what author", " what album",
        " what film", " what movie", " what character", " what alias",
        " what highway", " what length", " what does ",
        " what nationality", " what political party", " what star sign",
        " what is the current name", " what current name", " what scandal",
        " based in", " based out of", " called what", " also called what",
        " sister school", " in what new york ", " what nationality"
    ]
    has_question_cue = wh_start or any(q in t for q in question_cues) or t.endswith("?")

    # --------------------------------------------------
    # Factual / entity bridge cues
    # --------------------------------------------------
    factual_markers = [
        "named after", "based on", "first aired", "released in",
        "starred", "hosted", "directed", "written by", "performed",
        "played", "founded", "born", "died", "album", "film", "movie",
        "song", "episode", "season", "network", "actor", "actress",
        "author", "band", "magazine", "stadium", "city", "country",
        "county", "state", "university", "politician", "airport",
        "television series", "billboard", "nobel prize", "museum",
        "have in common", "what is the connection", "who funds",
        "who developed", "who invented", "who found", "who performed",
        "which industry", "which publishing company", "which musical",
        "which documentary", "which element", "which american-born",
        "older", "won more", "more members", "started first",
        "founded first", "born first", "same country", "same category",
        "what kind of track", "what kind of", "what type of",
        "grand slam", "hip hop", "football game", "new york county",
        "rock and roll hall of fame", "masterpiece theater", "track where",
        "head office", "chemical", "soluble", "what network", "what county",
        "what city", "what country", "what highway", "what age", "what year",
        "better known", "pioneered the use of", "what does", "who is older",
        "nationality", "political party", "star sign", "oil scandal",
        "famous for", "current name", "u.s highway", "us highway", "highway",
        "sister school", "nationality"
    ]
    factual_hits = count_phrase_hits(t, factual_markers)

    # --------------------------------------------------
    # Proper-noun density helps with factual bridge questions
    # --------------------------------------------------
    named_entity_chunks = re.findall(
        r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*",
        raw
    )
    ne_count = len(named_entity_chunks)

    # --------------------------------------------------
    # Early reject for arithmetic-style math stories
    # --------------------------------------------------
    if not wh_start and has_number_like and arithmetic_hits >= 3:
        return False

    if has_number_like and arithmetic_hits >= 4 and "what is the best answer for the question among these?" not in t:
        return False

    # --------------------------------------------------
    # Decision rules
    # --------------------------------------------------
    if starts_yesno and not (has_number_like and arithmetic_hits >= 3):
        return True

    if has_question_cue and factual_hits >= 1 and not (has_number_like and arithmetic_hits >= 3):
        return True

    if wh_start and ne_count >= 1 and arithmetic_hits <= 1:
        return True

    if has_question_cue and ne_count >= 2 and factual_hits >= 1 and not (has_number_like and arithmetic_hits >= 3):
        return True

    return False
    
def detect_common_sense_subtype(text: str) -> str:
    t = text.lower().strip()

    # --------------------------------------------------
    # 1) Multiple choice QA — must come FIRST
    # These questions have numbered options to pick from
    # --------------------------------------------------
    if re.search(r'what is the best answer.*among these', t) or \
       re.search(r'\n\s*0\)', text) or \
       re.search(r'\n\s*1\)', text):
        return "multiple_choice_qa"

    # --------------------------------------------------
    # 2) Context-grounded lookup
    # --------------------------------------------------
    if _contains_any(t, [
        "answer the question using the context",
        "using the context",
        "according to the context",
        "based on the context"
    ]):
        return "context_grounded_lookup"

    # --------------------------------------------------
    # 3) Shared attribute / connection
    # --------------------------------------------------
    shared_markers = [
        "what profession does",
        "have in common",
        "what is the connection between",
        "same type of work",
        "which industry do",
        "belong to",
        "are both",
        "is both",
        "both women's magazines",
        "both american",
        "both cocktails",
        "both american punk rock musicians"
    ]
    if _contains_any(t, shared_markers):
        return "shared_attribute_or_connection"

    # --------------------------------------------------
    # 4) "Is X or Y the [superlative]..." — comparison disguised as boolean
    # Must come BEFORE boolean check
    # --------------------------------------------------
    if re.search(r'\bis\b.+\bor\b', t) and _contains_any(t, [
        "largest", "smallest", "oldest", "newest", "first",
        "more", "most", "longer", "shorter", "bigger", "founded",
        "born", "older", "younger", "started", "owned"
    ]):
        return "comparison_resolution"

    # --------------------------------------------------
    # 5) Boolean plausibility
    # --------------------------------------------------
    boolean_starts = [
        "is ", "are ", "was ", "were ", "can ", "could ",
        "would ", "does ", "do ", "did ", "has ", "have ",
        "will "
    ]
    if any(t.startswith(prefix) for prefix in boolean_starts):
        return "boolean_plausibility"

    # --------------------------------------------------
    # 6) Comparison resolution
    # --------------------------------------------------
    comparison_markers = [
        " or ",
        "born first",
        "founded first",
        "started first",
        "older",
        "younger",
        "won more",
        "has more members",
        "same length as",
        "which one is owned by",
        "which magazine was started first",
        "which band was founded first",
        "who was born first",
        "which player won more",
        "which band has more members"
    ]
    if _contains_any(t, comparison_markers):
        return "comparison_resolution"

    # --------------------------------------------------
    # 7) Entity bridge lookup — default
    # --------------------------------------------------
    return "entity_bridge_lookup"
    
def detect_domain(text: str) -> str:
    if looks_like_planning(text):
        return "planning"
    elif looks_like_future_prediction(text):
        return "future_prediction"
    elif looks_like_coding(text):
        return "coding"
    elif looks_like_common_sense(text):
        return "common_sense"
    else:
        return "math"

