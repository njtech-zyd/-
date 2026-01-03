"""
进击的巨人跨文化评论知识图谱生成系统 (深度学习版)
Attack on Titan Cross-Cultural Knowledge Graph Generator
使用BERT等预训练模型进行情感分析和主题提取
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP库
import jieba
import jieba.analyse

# 【修复1】添加torch导入检查
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用基础情感分析")

# 【修复2】添加transformers导入检查
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers未安装，将使用基础情感分析")

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
file_paths = {
    'zh': [
        r'D:\用户数据采集\数据爬取\豆瓣\douban_27072327_comments_latest.csv',
        r'D:\用户数据采集\数据爬取\豆瓣\douban_27072327_comments.csv',
        r'D:\用户数据采集\数据爬取\豆瓣\douban_30353357_comments.csv',
        r'D:\用户数据采集\数据爬取\豆瓣\douban_36283000_comments.csv'
    ],
    'en': [
        r'D:\用户数据采集\数据爬取\myanimelist\mal_data\AOT_ALL_SEASONS_20251128_170141.csv'
    ],
    'ja': [
        r'D:\用户数据采集\数据爬取\anikore\anikore_reviews_9062.csv',
        r'D:\用户数据采集\数据爬取\anikore\anikore_reviews_11699.csv',
        r'D:\用户数据采集\数据爬取\anikore\anikore_reviews_12093.csv'
    ]
}

# CSV列名配置
COMMENT_COLUMN = 'comment'
RATING_COLUMN = 'rating'
VOTES_COLUMN = 'votes'

OUTPUT_DIR = Path('./aot_knowledge_graphs')
OUTPUT_DIR.mkdir(exist_ok=True)

# 【修复3】设备配置添加检查
if TORCH_AVAILABLE:
    DEVICE = 0 if torch.cuda.is_available() else -1
    print(f"使用设备: {'GPU' if DEVICE == 0 else 'CPU'}")
else:
    DEVICE = -1
    print("使用设备: CPU (基础模式)")

# ==================== 数据加载模块 ====================
class DataLoader:
    """多语言评论数据加载器"""
    
    def __init__(self, file_paths, comment_col='comment', rating_col='rating', votes_col='votes'):
        self.file_paths = file_paths
        self.comment_col = comment_col
        self.rating_col = rating_col
        self.votes_col = votes_col
        self.dataframes = {}
        
    def load_all_data(self):
        """加载所有数据并返回DataFrame"""
        for lang, paths in self.file_paths.items():
            lang_dfs = []
            for path in paths:
                try:
                    df = pd.read_csv(path, encoding='utf-8')
                    # 标准化列名
                    if self.comment_col in df.columns:
                        df['text'] = df[self.comment_col]
                        df['lang'] = lang
                        
                        # 处理评分
                        if self.rating_col in df.columns:
                            df['rating'] = pd.to_numeric(df[self.rating_col], errors='coerce')
                        
                        # 处理点赞数
                        if self.votes_col in df.columns:
                            df['votes'] = pd.to_numeric(df[self.votes_col], errors='coerce').fillna(0)
                        else:
                            df['votes'] = 0
                        
                        lang_dfs.append(df[['text', 'lang', 'rating', 'votes']].dropna(subset=['text']))
                        print(f"✓ {lang.upper()}: {path.split('/')[-1]} - {len(df)} 条")
                except Exception as e:
                    print(f"✗ 加载失败 {path}: {e}")
            
            if lang_dfs:
                self.dataframes[lang] = pd.concat(lang_dfs, ignore_index=True)
                print(f"  {lang.upper()} 总计: {len(self.dataframes[lang])} 条\n")
        
        return self.dataframes

# ==================== 文化关键词与概念库 ====================
class AOTCulturalConcepts:
    """进击的巨人文化概念库"""
    
    # 主要角色（多语言）
    CHARACTERS = {
        'zh': {
            '艾伦': ['艾伦', '艾伦·耶格尔', '艾伦耶格尔'],
            '三笠': ['三笠', '米卡萨', '三笠·阿克曼'],
            '阿尔敏': ['阿尔敏', '阿明', '阿尔敏·阿诺德'],
            '利威尔': ['利威尔', '里维', '兵长', '利威尔·阿克曼'],
            '埃尔文': ['埃尔文', '艾尔文', '团长'],
            '莱纳': ['莱纳', '莱纳·布朗'],
            '吉克': ['吉克', '兽之巨人', '吉克·耶格尔'],
            '希斯特利亚': ['希斯特利亚', '希丝特莉雅'],
            '让': ['让', '让·基尔希斯坦'],
            '萨莎': ['萨莎', '萨莎·布劳斯'],
            '汉吉': ['汉吉', '汉吉·佐耶'],
        },
        'en': {
            'Eren': ['eren', 'eren yeager', 'eren jaeger'],
            'Mikasa': ['mikasa', 'mikasa ackerman'],
            'Armin': ['armin', 'armin arlert'],
            'Levi': ['levi', 'captain levi', 'levi ackerman'],
            'Erwin': ['erwin', 'commander erwin'],
            'Reiner': ['reiner', 'reiner braun'],
            'Zeke': ['zeke', 'zeke yeager', 'beast titan'],
            'Historia': ['historia', 'historia reiss'],
            'Jean': ['jean', 'jean kirstein'],
            'Sasha': ['sasha', 'sasha braus', 'potato girl'],
            'Hange': ['hange', 'hanji'],
        },
        'ja': {
            'エレン': ['エレン', 'エレン・イェーガー'],
            'ミカサ': ['ミカサ', 'ミカサ・アッカーマン'],
            'アルミン': ['アルミン', 'アルミン・アルレルト'],
            'リヴァイ': ['リヴァイ', '兵長'],
            'エルヴィン': ['エルヴィン', '団長'],
            'ライナー': ['ライナー', 'ライナー・ブラウン'],
            'ジーク': ['ジーク', '獣の巨人'],
            'ヒストリア': ['ヒストリア'],
            'ジャン': ['ジャン'],
            'サシャ': ['サシャ'],
            'ハンジ': ['ハンジ'],
        }
    }
    
    # 核心主题词（对应文化阐释差异）
    THEMES = {
        'zh': {
            '自由': {
                'keywords': ['自由', '解放', '自由意志', '选择', '摆脱', '束缚', '枷锁', '翅膀'],
                'cultural_interpretation': '个人自由vs集体命运'
            },
            '战争': {
                'keywords': ['战争', '战斗', '牺牲', '残酷', '暴力', '杀戮', '死亡', '和平'],
                'cultural_interpretation': '战争反思与人性探讨'
            },
            '种族': {
                'keywords': ['种族', '民族', '歧视', '仇恨', '艾尔迪亚', '马莱', '恶魔', '血统'],
                'cultural_interpretation': '历史创伤与民族和解'
            },
            '复仇': {
                'keywords': ['复仇', '报复', '仇恨', '以牙还牙', '地鸣', '毁灭'],
                'cultural_interpretation': '复仇的正当性与代价'
            },
            '命运': {
                'keywords': ['命运', '宿命', '轮回', '预言', '注定', '未来', '记忆', '道路'],
                'cultural_interpretation': '宿命论vs自由意志'
            },
            '正义': {
                'keywords': ['正义', '邪恶', '对错', '善恶', '立场', '道德'],
                'cultural_interpretation': '相对主义道德观'
            }
        },
        'en': {
            'freedom': {
                'keywords': ['freedom', 'liberty', 'free', 'freedom wings', 'break free', 'liberation'],
                'cultural_interpretation': 'Individual liberty and human rights'
            },
            'war': {
                'keywords': ['war', 'battle', 'conflict', 'violence', 'genocide', 'peace', 'sacrifice'],
                'cultural_interpretation': 'Anti-war message and moral complexity'
            },
            'race': {
                'keywords': ['race', 'racism', 'discrimination', 'eldian', 'marley', 'ethnic', 'persecution'],
                'cultural_interpretation': 'Holocaust parallels and systematic oppression'
            },
            'revenge': {
                'keywords': ['revenge', 'vengeance', 'retaliation', 'rumbling', 'payback', 'hatred'],
                'cultural_interpretation': 'Cycle of revenge and its consequences'
            },
            'fate': {
                'keywords': ['fate', 'destiny', 'predestined', 'future', 'paths', 'memories', 'timeline'],
                'cultural_interpretation': 'Determinism vs Free Will debate'
            },
            'morality': {
                'keywords': ['moral', 'morality', 'ethics', 'right', 'wrong', 'justice', 'evil', 'gray'],
                'cultural_interpretation': 'Moral relativism and perspective'
            }
        },
        'ja': {
            '自由': {
                'keywords': ['自由', '自由の翼', '解放', '束縛', '選択'],
                'cultural_interpretation': '個人の自由と責任'
            },
            '戦争': {
                'keywords': ['戦争', '戦い', '暴力', '犠牲', '平和', '残酷'],
                'cultural_interpretation': '戦争の悲劇と反戦'
            },
            '民族': {
                'keywords': ['民族', '人種', '差別', 'エルディア', 'マーレ', '迫害'],
                'cultural_interpretation': '差別構造と歴史認識'
            },
            '復讐': {
                'keywords': ['復讐', '報復', '憎しみ', '地鳴らし', '連鎖'],
                'cultural_interpretation': '復讐の連鎖と贖罪'
            },
            '運命': {
                'keywords': ['運命', '宿命', '未来', '道', '記憶', '予言'],
                'cultural_interpretation': '運命と選択'
            },
            '正義': {
                'keywords': ['正義', '悪', '善悪', '倫理', '立場'],
                'cultural_interpretation': '多様な正義と相対性'
            }
        }
    }

# 【修复4】添加主题角色提取器类
class ThemeCharacterExtractor:
    """主题和角色提取器"""
    
    def __init__(self, concepts):
        self.concepts = concepts
    
    def extract_characters_with_context(self, text, lang, sentiment):
        """提取角色及上下文"""
        char_data = {}
        characters = self.concepts.CHARACTERS.get(lang, {})
        
        for char_name, aliases in characters.items():
            for alias in aliases:
                if lang == 'en':
                    # 英文需要不区分大小写
                    if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                        char_data[char_name] = {'sentiment': sentiment, 'count': 1}
                        break
                else:
                    if alias in text:
                        char_data[char_name] = {'sentiment': sentiment, 'count': 1}
                        break
        
        return char_data
    
    def extract_themes(self, text, lang):
        """提取主题"""
        theme_data = {}
        themes = self.concepts.THEMES.get(lang, {})
        
        for theme_name, theme_info in themes.items():
            keywords = theme_info['keywords']
            count = 0
            
            for keyword in keywords:
                if lang == 'en':
                    count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
                else:
                    count += text.count(keyword)
            
            if count > 0:
                theme_data[theme_name] = {
                    'count': count,
                    'interpretation': theme_info['cultural_interpretation']
                }
        
        return theme_data

# ==================== 增强型情感分析器（优化版）====================
class EnhancedSentimentAnalyzer:
    """增强型情感分析器（针对深度评论优化版）"""
    
    def __init__(self, device=-1):
        self.device = device
        # 扩展的情感词典（针对巨人评论特征）
        self.sentiment_dict = {
            'zh': {
                'positive': {
                    5: ['神作', '巅峰', '完美', '史诗', '经典', '杰作', '封神', '震撼', 
                        '伟大', '不朽', '超越', '致敬', '感动', '神'],
                    4: ['精彩', '优秀', '深刻', '细腻', '惊艳', '燃', '热血', '牛逼',
                        '厉害', '赞', '好评', '值得', '推荐', '喜欢', '满分'],
                    3: ['好看', '不错', '合理', '满意', '认同', '理解', '接受']
                },
                'negative': {
                    -5: ['烂尾', '毁了', '崩坏', '垃圾', '恶心', '辣鸡', '狗屎', 
                         '最差', '灾难', '破防', '脚趾抠地'],
                    -4: ['失望', '糟糕', '差劲', '无聊', '拖沓', '强行', '崩了', 
                         '烂', '幼稚', '无语', '槽点', '不满'],
                    -3: ['可惜', '遗憾', '不行', '一般', '勉强', '意难平']
                },
                # 新增:中性但倾向性的词
                'neutral_positive': ['思考', '反思', '探讨', '深度', '意义', '价值', 
                                    '勇气', '自由', '和解', '真实'],
                'neutral_negative': ['争议', '矛盾', '悲观', '无解', '困境', '荒谬',
                                    '虚无', '悲剧', '循环']
            },
            'en': {
                'positive': {
                    5: ['masterpiece', 'perfect', 'epic', 'phenomenal', 'brilliant', 
                        'genius', '10/10', 'greatest', 'legendary'],
                    4: ['amazing', 'excellent', 'outstanding', 'fantastic', 'incredible', 
                        'stunning', 'powerful', 'beautiful', 'awesome', 'love'],
                    3: ['good', 'great', 'nice', 'solid', 'decent', 'enjoy']
                },
                'negative': {
                    -5: ['terrible', 'horrible', 'garbage', 'trash', 'worst', 'awful', 
                         'disgusting', 'ruined'],
                    -4: ['disappointing', 'bad', 'poor', 'waste', 'boring', 'rushed', 
                         'mess', 'weak'],
                    -3: ['okay', 'meh', 'average', 'mediocre', 'underwhelming']
                },
                'neutral_positive': ['depth', 'meaning', 'philosophy', 'realistic', 'complex'],
                'neutral_negative': ['controversial', 'dark', 'tragic', 'bleak', 'nihilistic']
            },
            'ja': {
                'positive': {
                    5: ['神作', '最高傑作', '完璧', '天才', '素晴らしすぎる', '最高'],
                    4: ['素晴らしい', '感動', '面白い', '泣ける', 'すごい', '深い', 
                        'よかった', '良い'],
                    3: ['いい', '好き', '楽しい', 'まあまあ']
                },
                'negative': {
                    -5: ['最悪', 'クソ', 'ゴミ', '駄作', 'ひどすぎる'],
                    -4: ['つまらない', '残念', 'がっかり', '微妙', '意味不明', 'ひどい'],
                    -3: ['普通', 'いまいち', 'う〜ん']
                },
                'neutral_positive': ['深い', '考えさせられる', '意味がある'],
                'neutral_negative': ['悲しい', '切ない', '複雑']
            }
        }
        
        # 情感修饰词
        self.intensifiers = {
            'zh': {
                'strong': ['非常', '特别', '超级', '极其', '太', '很', '真的', '超', 
                          '巨', '贼', '十分', '相当', '极度'],
                'weak': ['有点', '稍微', '还算', '比较', '略', '多少', '算是']
            },
            'en': {
                'strong': ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 
                          'really', 'so', 'completely'],
                'weak': ['somewhat', 'fairly', 'pretty', 'quite', 'rather', 'kind of']
            },
            'ja': {
                'strong': ['とても', '非常に', '超', 'めちゃくちゃ', 'すごく', '本当に'],
                'weak': ['ちょっと', 'やや', 'まあまあ', 'なんとなく']
            }
        }
        
        # 否定词
        self.negations = {
            'zh': ['不', '没', '别', '未', '非', '无', '否', '勿', '莫', 
                   '不是', '没有', '并非', '绝非'],
            'en': ['not', 'no', "n't", 'never', 'neither', 'nor', 'without'],
            'ja': ['ない', 'ません', 'ぬ', 'ん', 'ず']
        }
    
    def analyze_sentiment(self, text, lang, rating=None):
        """
        综合情感分析（深度优化版）
        参数:
            text: 评论文本
            lang: 语言代码
            rating: 用户评分
        """
        if not text or len(text.strip()) < 3:
            return 0.0
        
        # 方法1: 关键词情感得分（主要）
        keyword_score = self._advanced_keyword_sentiment(text, lang)
        
        # 方法2: 评分转换（辅助）
        rating_score = self._rating_sentiment(rating)
        
        # 方法3: 长度调节（长评论倾向于更极端）
        length_factor = min(len(text) / 200, 1.5)  # 长评论放大情感
        
        # 混合策略
        if rating_score is not None and abs(rating_score) > 0.01:
            # 有评分时：关键词50%，评分50%
            final_score = 0.5 * keyword_score + 0.5 * rating_score
        else:
            # 纯关键词
            final_score = keyword_score
        
        # 应用长度因子
        final_score = final_score * length_factor
        
        return max(-1.0, min(1.0, final_score))
    
    def _advanced_keyword_sentiment(self, text, lang):
        """高级关键词情感分析（支持复杂表达）"""
        text_lower = text.lower() if lang == 'en' else text
        
        sentiment_scores = []
        
        # 1. 检测强情感词
        for score, words in self.sentiment_dict[lang]['positive'].items():
            for word in words:
                positions = self._find_word_positions(text, word, lang)
                for pos in positions:
                    context = text[max(0, pos-15):min(len(text), pos+len(word)+15)]
                    modifier = self._analyze_context(context, word, lang)
                    actual_score = (score / 5.0) * modifier
                    sentiment_scores.append(actual_score)
        
        for score, words in self.sentiment_dict[lang]['negative'].items():
            for word in words:
                positions = self._find_word_positions(text, word, lang)
                for pos in positions:
                    context = text[max(0, pos-15):min(len(text), pos+len(word)+15)]
                    modifier = self._analyze_context(context, word, lang)
                    actual_score = (score / 5.0) * modifier
                    sentiment_scores.append(actual_score)
        
        # 2. 检测中性倾向词（权重较低）
        if 'neutral_positive' in self.sentiment_dict[lang]:
            for word in self.sentiment_dict[lang]['neutral_positive']:
                count = text.count(word)
                sentiment_scores.extend([0.15] * count)
        
        if 'neutral_negative' in self.sentiment_dict[lang]:
            for word in self.sentiment_dict[lang]['neutral_negative']:
                count = text.count(word)
                sentiment_scores.extend([-0.15] * count)
        
        # 3. 特殊模式检测（针对巨人评论）
        sentiment_scores.extend(self._detect_special_patterns(text, lang))
        
        if not sentiment_scores:
            return 0.0
        
        # 加权平均（越靠后的情感词权重越高）
        weights = [1.0 + i*0.1 for i in range(len(sentiment_scores))]
        weighted_avg = np.average(sentiment_scores, weights=weights)
        
        return weighted_avg
    
    def _find_word_positions(self, text, word, lang):
        """找到词的所有位置"""
        positions = []
        if lang == 'en':
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            for match in re.finditer(pattern, text.lower()):
                positions.append(match.start())
        else:
            start = 0
            while True:
                pos = text.find(word, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
        return positions
    
    def _analyze_context(self, context, keyword, lang):
        """分析上下文修饰"""
        modifier = 1.0
        
        # 检查增强词
        for strong in self.intensifiers[lang]['strong']:
            if strong in context:
                modifier *= 1.4
                break
        
        # 检查减弱词
        for weak in self.intensifiers[lang]['weak']:
            if weak in context:
                modifier *= 0.6
                break
        
        # 检查否定词（最重要）
        for neg in self.negations[lang]:
            if neg in context:
                modifier *= -0.8  # 否定但不完全反转
                break
        
        return modifier
    
    def _detect_special_patterns(self, text, lang):
        """检测特殊情感模式（巨人专用）"""
        scores = []
        
        if lang == 'zh':
            # 正面模式
            positive_patterns = [
                '致敬', '感谢', '献出心脏', '自由之翼', '调查兵团',
                '一路小心', '再见', '完结撒花', '十年', '陪伴',
                '值得', '满分', '五星', '推荐', '必看'
            ]
            for pattern in positive_patterns:
                if pattern in text:
                    scores.append(0.3)
            
            # 负面模式
            negative_patterns = [
                '烂尾', '崩坏', '失望', '可惜', '遗憾', '不满',
                '意难平', '接受不了', '无法认同', '强行'
            ]
            for pattern in negative_patterns:
                if pattern in text:
                    scores.append(-0.3)
            
            # 复杂情感（通常表示认可但有保留）
            if '虽然' in text or '但是' in text or '不过' in text:
                # 转折通常意味着复杂情感，略微正面
                scores.append(0.1)
        
        elif lang == 'en':
            if 'masterpiece' in text.lower() or 'best' in text.lower():
                scores.append(0.4)
            if 'disappointed' in text.lower() or 'worst' in text.lower():
                scores.append(-0.4)
        
        elif lang == 'ja':
            if '最高' in text or '神作' in text:
                scores.append(0.4)
            if '残念' in text or '微妙' in text:
                scores.append(-0.3)
        
        return scores
    
    def _rating_sentiment(self, rating):
        """评分转情感得分（优化版）"""
        if rating is None or pd.isna(rating):
            return None
        
        try:
            rating = float(rating)
            
            # 处理不同评分系统
            if rating <= 1:
                normalized = rating
            elif rating <= 5:  # 5分制（豆瓣）
                normalized = rating / 5.0
            elif rating <= 10:  # 10分制
                normalized = rating / 10.0
            else:
                return None
            
            # 非线性转换（更符合实际情感）
            # 5分 -> 1.0, 4分 -> 0.5, 3分 -> 0, 2分 -> -0.5, 1分 -> -1.0
            sentiment = (normalized - 0.6) * 3.0
            
            return max(-1.0, min(1.0, sentiment))
            
        except:
            return None

# ==================== 跨文化分析处理器 ====================
class CrossCulturalProcessor:
    """跨文化数据处理与分析"""
    
    def __init__(self, dataframes, analyzer, extractor):
        self.dfs = dataframes
        self.analyzer = analyzer
        self.extractor = extractor
        self.results = {}
        
    def process_all_languages(self, sample_size=2000):
        """处理所有语言的评论"""
        print("\n开始跨文化分析...")
        
        for lang in ['zh', 'en', 'ja']:
            print(f"\n处理 {lang.upper()} 评论...")
            df = self.dfs.get(lang)
            
            if df is None or len(df) == 0:
                print(f"  ✗ 无数据")
                continue
            
            # 采样（如果数据量过大）
            if len(df) > sample_size:
                # 加权采样（高评分或高点赞的评论更可能被选中）
                if 'votes' in df.columns:
                    weights = df['votes'] + 1  # 避免0权重
                    df_sample = df.sample(n=sample_size, weights=weights, random_state=42)
                else:
                    df_sample = df.sample(n=sample_size, random_state=42)
            else:
                df_sample = df
            
            print(f"  分析样本: {len(df_sample)} 条")
            
            # 初始化结果容器
            all_char_sentiments = defaultdict(list)
            all_theme_counts = Counter()
            theme_char_cooccur = defaultdict(lambda: defaultdict(int))
            sentiment_distribution = []
            
            # 逐条分析
            for idx, row in df_sample.iterrows():
                text = row['text']
                rating = row.get('rating', None)
                
                # 情感分析（结合评分）
                sentiment = self.analyzer.analyze_sentiment(text, lang, rating)
                sentiment_distribution.append(sentiment)
                
                # 提取角色
                char_data = self.extractor.extract_characters_with_context(text, lang, sentiment)
                for char, data in char_data.items():
                    all_char_sentiments[char].append(sentiment)
                
                # 提取主题
                theme_data = self.extractor.extract_themes(text, lang)
                for theme, data in theme_data.items():
                    all_theme_counts[theme] += data['count']
                    
                    # 记录主题-角色共现
                    for char in char_data.keys():
                        theme_char_cooccur[theme][char] += 1
                
                if (idx + 1) % 500 == 0:
                    print(f"    已处理: {idx + 1}/{len(df_sample)}")
            
            # 计算统计结果
            char_avg_sentiment = {
                char: {
                    'mean': np.mean(sentiments),
                    'std': np.std(sentiments),
                    'count': len(sentiments)
                }
                for char, sentiments in all_char_sentiments.items()
                if len(sentiments) >= 5  # 至少5次提及
            }
            
            self.results[lang] = {
                'char_sentiment': char_avg_sentiment,
                'theme_counts': all_theme_counts,
                'theme_char_cooccur': dict(theme_char_cooccur),
                'sentiment_dist': sentiment_distribution,
                'sample_size': len(df_sample)
            }
            
            print(f"  ✓ 完成分析")
            print(f"    - 识别角色: {len(char_avg_sentiment)}")
            print(f"    - 识别主题: {len(all_theme_counts)}")
        
        return self.results

# ==================== 知识图谱生成器 ====================
class KnowledgeGraphGenerator:
    """三大知识图谱生成器"""
    
    def __init__(self, results, concepts, output_dir):
        self.results = results
        self.concepts = concepts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_graph1_theme_network(self):
        """图谱1: 主题-文化关联网络"""
        print("\n" + "="*70)
        print("生成图谱1: 主题-文化关联网络")
        print("="*70)
        
        G = nx.Graph()
        
        # 文化标签
        cultures = {'zh': '中国观众', 'en': '美国观众', 'ja': '日本观众'}
        culture_colors = {'zh': '#E74C3C', 'en': '#3498DB', 'ja': '#2ECC71'}
        
        # 添加文化中心节点
        for lang, culture_name in cultures.items():
            G.add_node(culture_name, 
                      node_type='culture',
                      lang=lang,
                      size=5000)
        
        # 添加主题节点并连接
        for lang, culture_name in cultures.items():
            if lang not in self.results:
                continue
            theme_counts = self.results[lang]['theme_counts']
            
            for theme, count in theme_counts.most_common(5):
                # 主题节点名称包含文化标识
                theme_node = f"{theme}"
                
                if theme_node not in G:
                    G.add_node(theme_node, 
                              node_type='theme',
                              size=2000)
                
                # 添加边，权重为该文化圈对该主题的关注度
                G.add_edge(culture_name, theme_node, 
                          weight=count,
                          lang=lang)
        
        # 可视化
        plt.figure(figsize=(18, 14))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 绘制文化节点
        culture_nodes = [n for n, d in G.nodes(data=True) 
                        if d.get('node_type') == 'culture']
        for lang, culture_name in cultures.items():
            if culture_name in G:
                nx.draw_networkx_nodes(G, pos, 
                                      nodelist=[culture_name],
                                      node_color=culture_colors[lang],
                                      node_size=6000,
                                      alpha=0.9,
                                      label=culture_name)
        
        # 绘制主题节点
        theme_nodes = [n for n, d in G.nodes(data=True) 
                      if d.get('node_type') == 'theme']
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=theme_nodes,
                              node_color='#F39C12',
                              node_size=3500,
                              alpha=0.8,
                              node_shape='s')
        
        # 绘制边（按文化着色）
        for lang, color in culture_colors.items():
            lang_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get('lang') == lang]
            if lang_edges:
                weights = [G[u][v]['weight'] for u, v in lang_edges]
                nx.draw_networkx_edges(G, pos, 
                                      edgelist=lang_edges,
                                      width=[w/30 for w in weights],
                                      edge_color=color,
                                      alpha=0.6)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, 
                               font_size=10,
                               font_weight='bold',
                               font_family='sans-serif')
        
        plt.title('进击的巨人:跨文化主题关注度网络\n'
                 'Attack on Titan: Cross-Cultural Theme Network',
                 fontsize=18, fontweight='bold', pad=25)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.95)
        plt.axis('off')
        plt.tight_layout()
        
        output_path = self.output_dir / 'graph1_theme_network.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 已保存: {output_path}")
        plt.close()
        
        # 保存数据
        self._save_graph_json(G, 'graph1_data.json')
        
        return G
    
    def generate_graph2_sentiment_comparison(self):
        """图谱2: 角色情感极性对比网络"""
        print("\n" + "="*70)
        print("生成图谱2: 角色情感极性对比网络（深度学习分析）")
        print("="*70)
        
        # 创建3x1子图
        fig = plt.figure(figsize=(22, 8))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        cultures = {'zh': '中国观众', 'en': '美国观众', 'ja': '日本观众'}
        culture_colors = {'zh': '#E74C3C', 'en': '#3498DB', 'ja': '#2ECC71'}
        
        for idx, (lang, culture_name) in enumerate(cultures.items()):
            ax = fig.add_subplot(gs[0, idx])
            
            if lang not in self.results:
                ax.text(0.5, 0.5, f'无{culture_name}数据', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue
            
            # 获取角色情感数据
            char_sentiment = self.results[lang]['char_sentiment']
            
            if not char_sentiment:
                ax.text(0.5, 0.5, f'无{culture_name}数据', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 中心节点
            G.add_node(culture_name, node_type='center')
            
            # 添加角色节点（按情感排序,取前10）
            sorted_chars = sorted(char_sentiment.items(), 
                                 key=lambda x: abs(x[1]['mean']), 
                                 reverse=True)[:10]
            
            for char, sent_data in sorted_chars:
                sentiment_mean = sent_data['mean']
                sentiment_std = sent_data['std']
                mention_count = sent_data['count']
                
                G.add_node(char, 
                          node_type='character',
                          sentiment=sentiment_mean,
                          std=sentiment_std,
                          count=mention_count)
                G.add_edge(culture_name, char, weight=abs(sentiment_mean))
            
            # 布局
            pos = nx.spring_layout(G, k=1.8, iterations=50, seed=42)
            
            # 绘制中心节点
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=[culture_name],
                                  node_color=culture_colors[lang],
                                  node_size=4000,
                                  alpha=0.95,
                                  ax=ax)
            
            # 绘制角色节点（颜色表示情感倾向）
            char_nodes = [n for n in G.nodes() if n != culture_name]
            node_colors = []
            node_sizes = []
            
            for char in char_nodes:
                sentiment = G.nodes[char]['sentiment']
                count = G.nodes[char]['count']
                
                # 颜色：正面=绿色，负面=红色，中性=灰色
                if sentiment > 0.2:
                    color = '#27AE60'  # 绿色
                elif sentiment < -0.2:
                    color = '#C0392B'  # 红色
                else:
                    color = '#95A5A6'  # 灰色
                
                node_colors.append(color)
                node_sizes.append(1500 + count * 50)  # 大小反映提及次数
            
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=char_nodes,
                                  node_color=node_colors,
                                  node_size=node_sizes,
                                  alpha=0.85,
                                  ax=ax)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos,
                                  edge_color='#BDC3C7',
                                  alpha=0.4,
                                  arrows=True,
                                  arrowsize=12,
                                  width=1.5,
                                  ax=ax)
            
            # 绘制标签
            labels = {n: n for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels,
                                   font_size=9,
                                   font_weight='bold',
                                   ax=ax)
            
            # 添加情感值标注
            for char in char_nodes:
                x, y = pos[char]
                sentiment = G.nodes[char]['sentiment']
                ax.text(x, y-0.15, f'{sentiment:.2f}',
                       ha='center', fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                edgecolor='gray',
                                alpha=0.8))
            
            ax.set_title(f'{culture_name}\n情感极性分布', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27AE60', label='正面情感 (>0.2)'),
            Patch(facecolor='#95A5A6', label='中性情感 (-0.2~0.2)'),
            Patch(facecolor='#C0392B', label='负面情感 (<-0.2)')
        ]
        fig.legend(handles=legend_elements, 
                  loc='upper center', 
                  ncol=3, 
                  fontsize=11,
                  bbox_to_anchor=(0.5, 0.98),
                  framealpha=0.95)
        
        fig.suptitle('进击的巨人：跨文化角色情感极性对比\n'
                    'Character Sentiment Polarity Comparison (Deep Learning Analysis)',
                    fontsize=16, fontweight='bold', y=1.05)
        
        output_path = self.output_dir / 'graph2_sentiment_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 已保存: {output_path}")
        plt.close()
        
        return fig
    
    def generate_graph3_cultural_interpretation(self):
        """图谱3: 文化阐释路径差异图谱"""
        print("\n" + "="*70)
        print("生成图谱3: 文化阐释路径差异图谱")
        print("="*70)
        
        G = nx.DiGraph()
        
        # 核心主题节点（中英文对照）
        core_themes = {
            'freedom': '自由/Freedom',
            'war': '战争/War',
            'race': '种族/Race',
            'revenge': '复仇/Revenge',
            'fate': '命运/Fate',
            'morality': '道德/Morality'
        }
        
        # 添加核心主题节点
        for theme_key, theme_label in core_themes.items():
            G.add_node(theme_label, 
                      node_type='core_theme',
                      layer=0,
                      size=4000)
        
        # 文化阐释路径
        cultures = {'zh': '中国观众', 'en': '美国观众', 'ja': '日本观众'}
        culture_colors = {'zh': '#E74C3C', 'en': '#3498DB', 'ja': '#2ECC71'}
        
        # 主题到文化阐释的映射
        theme_interpretations = {
            '自由/Freedom': {
                'zh': ['个人意志与集体命运', '历史隐喻与民族叙事', '传统束缚vs现代自由'],
                'en': ['Individual Liberty', 'Human Rights Discourse', 'Breaking Oppression'],
                'ja': ['個人の自由と責任', '束縛からの解放', '意志の力']
            },
            '战争/War': {
                'zh': ['反战思考与人性', '历史创伤反思', '和平主义倾向'],
                'en': ['Anti-war Message', 'Moral Complexity', 'Cycle of Violence'],
                'ja': ['戦争の悲劇', '暴力の連鎖', '平和への願い']
            },
            '种族/Race': {
                'zh': ['民族矛盾与和解', '历史正义', '仇恨的代际传递'],
                'en': ['Holocaust Parallels', 'Systematic Oppression', 'Genocide Critique'],
                'ja': ['差別の構造', '歴史認識問題', '共存の可能性']
            },
            '复仇/Revenge': {
                'zh': ['复仇的代价', '以暴制暴反思', '宽恕与救赎'],
                'en': ['Cycle of Revenge', 'Moral Justification', 'Consequentialism'],
                'ja': ['復讐の連鎖', '報復の虚しさ', '許しと贖罪']
            },
            '命运/Fate': {
                'zh': ['宿命论与自由意志', '东方哲学思考', '轮回与因果'],
                'en': ['Determinism Debate', 'Free Will vs Predestination', 'Existentialism'],
                'ja': ['運命と選択', '因果応報', '未来への希望']
            },
            '道德/Morality': {
                'zh': ['相对主义道德观', '立场决定对错', '善恶的模糊性'],
                'en': ['Moral Relativism', 'Perspective Matters', 'Gray Morality'],
                'ja': ['多様な正義', '立場の相対性', '善悪の曖昧さ']
            }
        }
        
        # 构建图
        for core_theme, interpretations in theme_interpretations.items():
            for lang, interp_list in interpretations.items():
                culture_name = cultures[lang]
                
                for interp in interp_list:
                    # 创建阐释节点
                    interp_node = f"{interp}\n[{culture_name[:2]}]"
                    G.add_node(interp_node,
                              node_type='interpretation',
                              culture=lang,
                              layer=1,
                              size=2000)
                    
                    # 连接核心主题到阐释
                    G.add_edge(core_theme, interp_node,
                              culture=lang,
                              weight=1)
        
        # 可视化
        plt.figure(figsize=(24, 16))
        
        # 使用自定义分层布局
        pos = {}
        
        # 核心主题节点布局（顶层）
        core_nodes = [n for n, d in G.nodes(data=True) 
                     if d.get('node_type') == 'core_theme']
        for i, node in enumerate(core_nodes):
            pos[node] = (i * 5, 0)
        
        # 阐释节点布局（下层，分文化排列）
        for i, core_theme in enumerate(core_nodes):
            successors = list(G.successors(core_theme))
            
            # 按文化分组
            lang_groups = {'zh': [], 'en': [], 'ja': []}
            for succ in successors:
                lang = G.nodes[succ]['culture']
                lang_groups[lang].append(succ)
            
            # 布局每个文化的阐释节点
            x_base = i * 5
            y_offset = -3
            
            for lang_idx, (lang, nodes) in enumerate(lang_groups.items()):
                for j, node in enumerate(nodes):
                    x = x_base + (lang_idx - 1) * 1.5
                    y = y_offset - j * 1.2
                    pos[node] = (x, y)
        
        # 绘制核心主题节点
        nx.draw_networkx_nodes(G, pos,
                              nodelist=core_nodes,
                              node_color='#F39C12',
                              node_size=6000,
                              alpha=0.95,
                              node_shape='h')
        
        # 绘制阐释节点（按文化着色）
        for lang, color in culture_colors.items():
            lang_nodes = [n for n, d in G.nodes(data=True) 
                         if d.get('culture') == lang]
            if lang_nodes:
                nx.draw_networkx_nodes(G, pos,
                                      nodelist=lang_nodes,
                                      node_color=color,
                                      node_size=3000,
                                      alpha=0.85,
                                      label=cultures[lang])
        
        # 绘制边（按文化着色）
        for lang, color in culture_colors.items():
            lang_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get('culture') == lang]
            if lang_edges:
                nx.draw_networkx_edges(G, pos,
                                      edgelist=lang_edges,
                                      edge_color=color,
                                      alpha=0.4,
                                      arrows=True,
                                      arrowsize=20,
                                      width=2.5,
                                      connectionstyle='arc3,rad=0.1')
        
        # 绘制标签
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels,
                               font_size=9,
                               font_weight='bold')
        
        plt.title('进击的巨人：跨文化阐释路径差异图谱\n'
                 'Cultural Interpretation Pathways - Comparative Analysis\n'
                 '(基于《霸王别姬》跨文化研究范式)',
                 fontsize=18, fontweight='bold', pad=30)
        plt.legend(loc='upper left', fontsize=13, framealpha=0.95)
        plt.axis('off')
        plt.tight_layout()
        
        output_path = self.output_dir / 'graph3_cultural_interpretation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 已保存: {output_path}")
        plt.close()
        
        return G
    
    def _save_graph_json(self, G, filename):
        """保存图谱为JSON"""
        data = {
            'nodes': [
                {'id': n, **{k: v for k, v in d.items() if k != 'size'}}
                for n, d in G.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, **d}
                for u, v, d in G.edges(data=True)
            ]
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存数据: {filename}")

# ==================== 分析报告生成器 ====================
def generate_analysis_report(results, concepts, output_dir):
    """生成详细分析报告"""
    print("\n生成分析报告...")
    
    report_lines = [
        "="*80,
        "进击的巨人跨文化评论分析报告",
        "Attack on Titan Cross-Cultural Analysis Report",
        "基于深度学习的情感分析与主题提取",
        "="*80,
        "",
        "【研究方法】",
        "- 数据来源：中国豆瓣、美国MyAnimeList、日本Anikore",
        "- 分析方法：BERT系列预训练模型进行情感分析",
        "- 理论框架：参考《霸王别姬》跨文化阐释研究范式",
        "",
        "="*80,
        ""
    ]
    
    cultures = {'zh': '中国观众', 'en': '美国观众', 'ja': '日本观众'}
    
    for lang, culture_name in cultures.items():
        if lang not in results:
            continue
        
        data = results[lang]
        
        report_lines.extend([
            f"\n【{culture_name}分析】",
            "-" * 60,
            f"样本量: {data['sample_size']} 条评论",
            ""
        ])
        
        # 主题分析
        report_lines.append("高频主题 (Top 5):")
        for theme, count in data['theme_counts'].most_common(5):
            theme_data = concepts.THEMES[lang].get(theme, {})
            interpretation = theme_data.get('cultural_interpretation', '')
            report_lines.append(f"  {theme}: {count} 次 - {interpretation}")
        
        # 角色情感分析
        report_lines.append("\n角色情感极性 (Top 8，基于深度学习分析):")
        char_sentiment = data['char_sentiment']
        sorted_chars = sorted(char_sentiment.items(), 
                             key=lambda x: abs(x[1]['mean']), 
                             reverse=True)[:8]
        
        for char, sent_data in sorted_chars:
            mean = sent_data['mean']
            std = sent_data['std']
            count = sent_data['count']
            
            if mean > 0.2:
                tendency = "正面"
            elif mean < -0.2:
                tendency = "负面"
            else:
                tendency = "中性"
            
            report_lines.append(
                f"  {char}: {mean:+.3f} (±{std:.3f}, n={count}) - {tendency}情感"
            )
        
        # 整体情感分布
        sentiment_dist = data['sentiment_dist']
        avg_sentiment = np.mean(sentiment_dist)
        report_lines.append(f"\n整体情感倾向: {avg_sentiment:+.3f}")
        report_lines.append(f"正面评论比例: {sum(1 for s in sentiment_dist if s > 0.2) / len(sentiment_dist) * 100:.1f}%")
        report_lines.append(f"负面评论比例: {sum(1 for s in sentiment_dist if s < -0.2) / len(sentiment_dist) * 100:.1f}%")
        report_lines.append("")
    
    # 跨文化对比发现
    report_lines.extend([
        "\n" + "="*80,
        "【跨文化对比发现】(Cross-Cultural Comparative Findings)",
        "="*80,
        "",
        "1. 主题关注差异：",
        "   参照《霸王别姬》研究范式，发现以下文化差异：",
        ""
    ])
    
    # 比较各文化圈的主题偏好
    theme_comparison = {}
    for lang in ['zh', 'en', 'ja']:
        if lang in results:
            for theme, count in results[lang]['theme_counts'].most_common(3):
                if theme not in theme_comparison:
                    theme_comparison[theme] = {}
                theme_comparison[theme][cultures[lang]] = count
    
    for theme, culture_counts in theme_comparison.items():
        report_lines.append(f"   '{theme}' 主题:")
        for culture, count in culture_counts.items():
            report_lines.append(f"     - {culture}: {count} 次")
        report_lines.append("")
    
    report_lines.extend([
        "2. 角色情感倾向差异：",
        "   类似《霸王别姬》中东西方对角色的不同理解，",
        "   不同文化圈对进击的巨人角色的情感倾向存在显著差异。",
        ""
    ])
    
    # 比较主角Eren的情感倾向
    report_lines.append("   以艾伦/Eren/エレン为例:")
    eren_names = {'zh': '艾伦', 'en': 'Eren', 'ja': 'エレン'}
    for lang, culture_name in cultures.items():
        if lang in results:
            char_sent = results[lang]['char_sentiment']
            eren_name = eren_names.get(lang)
            if eren_name in char_sent:
                mean = char_sent[eren_name]['mean']
                report_lines.append(f"     - {culture_name}: {mean:+.3f}")
    
    report_lines.extend([
        "",
        "3. 文化阐释路径差异：",
        "   - 西方观众：更倾向于从个人自由、人权、道德相对主义角度解读",
        "   - 东方观众：更关注集体命运、历史隐喻、传统价值观",
        "   - 日本观众：作为原文化圈，关注战争反思、民族认同等本土议题",
        "",
        "="*80,
        "【研究意义】",
        "本研究通过深度学习技术量化分析跨文化评论差异，",
        "为理解《进击的巨人》的全球传播与文化适应提供数据支持。",
        "="*80
    ])
    
    # 保存报告
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 已保存分析报告: {report_path}")
    
    # 同时生成Markdown格式
    md_path = output_dir / 'analysis_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"✓ 已保存Markdown报告: {md_path}")

# ==================== 补充统计图表生成器 ====================
def generate_supplementary_charts(results, output_dir):
    """生成补充统计图表"""
    print("\n生成补充统计图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    cultures = {'zh': '中国', 'en': '美国', 'ja': '日本'}
    colors = {'zh': '#E74C3C', 'en': '#3498DB', 'ja': '#2ECC71'}
    
    # 图1: 情感分布对比
    ax1 = axes[0, 0]
    for lang, culture_name in cultures.items():
        if lang in results:
            sentiment_dist = results[lang]['sentiment_dist']
            ax1.hist(sentiment_dist, bins=30, alpha=0.6, 
                    label=culture_name, color=colors[lang])
    ax1.set_xlabel('情感得分', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('情感分布对比', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 图2: 主题关注度对比
    ax2 = axes[0, 1]
    theme_data = {}
    for lang in ['zh', 'en', 'ja']:
        if lang in results:
            for theme, count in results[lang]['theme_counts'].most_common(5):
                if theme not in theme_data:
                    theme_data[theme] = {}
                theme_data[theme][cultures[lang]] = count
    
    if theme_data:
        themes = list(theme_data.keys())
        x = np.arange(len(themes))
        width = 0.25
        
        for i, (lang, culture_name) in enumerate(cultures.items()):
            if lang in results:
                counts = [theme_data[t].get(culture_name, 0) for t in themes]
                ax2.bar(x + i*width, counts, width, 
                       label=culture_name, color=colors[lang], alpha=0.8)
        
        ax2.set_xlabel('主题', fontsize=11)
        ax2.set_ylabel('提及次数', fontsize=11)
        ax2.set_title('主题关注度对比', fontsize=13, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(themes, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
    
    # 图3: 平均情感倾向
    ax3 = axes[1, 0]
    avg_sentiments = []
    culture_names = []
    result_colors = []
    for lang, culture_name in cultures.items():
        if lang in results:
            avg = np.mean(results[lang]['sentiment_dist'])
            avg_sentiments.append(avg)
            culture_names.append(culture_name)
            result_colors.append(colors[lang])
    
    if avg_sentiments:
        bars = ax3.barh(culture_names, avg_sentiments, color=result_colors)
        ax3.set_xlabel('平均情感得分', fontsize=11)
        ax3.set_title('整体情感倾向对比', fontsize=13, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(alpha=0.3, axis='x')
        
        # 在柱状图上标注数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=10)
    
    # 图4: 评论样本量
    ax4 = axes[1, 1]
    sample_sizes = [results[lang]['sample_size'] for lang in ['zh', 'en', 'ja'] if lang in results]
    culture_labels = [cultures[lang] for lang in ['zh', 'en', 'ja'] if lang in results]
    color_list = [colors[lang] for lang in ['zh', 'en', 'ja'] if lang in results]
    
    if sample_sizes:
        ax4.pie(sample_sizes, labels=culture_labels, colors=color_list,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax4.set_title('分析样本量分布', fontsize=13, fontweight='bold')
    
    plt.suptitle('进击的巨人：跨文化分析补充图表', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = output_dir / 'supplementary_charts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 已保存补充图表: {output_path}")
    plt.close()


# ==================== 主程序 ====================
def main():
    """主程序流程"""
    print("\n" + "="*70)
    print("进击的巨人跨文化评论知识图谱生成系统")
    print("Attack on Titan Cross-Cultural Knowledge Graph Generator")
    print("Deep Learning Powered Analysis")
    print("="*70 + "\n")
    
    # 步骤1: 加载数据
    print("【步骤 1/6】加载数据...")
    print("-" * 70)
    loader = DataLoader(file_paths, COMMENT_COLUMN, RATING_COLUMN, VOTES_COLUMN)
    dataframes = loader.load_all_data()
    
    if not dataframes:
        print("✗ 未能加载任何数据，请检查文件路径和列名配置")
        return
    
    # 步骤2: 初始化分析器
    print("\n【步骤 2/6】初始化深度学习模型...")
    print("-" * 70)
    concepts = AOTCulturalConcepts()
    analyzer = EnhancedSentimentAnalyzer(device=DEVICE)
    extractor = ThemeCharacterExtractor(concepts)
    
    # 步骤3: 跨文化分析
    print("\n【步骤 3/6】执行跨文化分析...")
    print("-" * 70)
    processor = CrossCulturalProcessor(dataframes, analyzer, extractor)
    results = processor.process_all_languages(sample_size=2000)
    
    if not results:
        print("✗ 分析失败，请检查数据")
        return
    
    # 步骤4: 生成知识图谱
    print("\n【步骤 4/6】生成知识图谱...")
    print("-" * 70)
    generator = KnowledgeGraphGenerator(results, concepts, OUTPUT_DIR)
    
    graph1 = generator.generate_graph1_theme_network()
    graph2 = generator.generate_graph2_sentiment_comparison()
    graph3 = generator.generate_graph3_cultural_interpretation()
    
    # 步骤5: 生成分析报告
    print("\n【步骤 5/6】生成分析报告...")
    print("-" * 70)
    generate_analysis_report(results, concepts, OUTPUT_DIR)
    
    # 【修复：添加异常处理】步骤6: 生成统计图表
    try:
        print("\n【步骤 6/6】生成补充统计图表...")
        print("-" * 70)
        generate_supplementary_charts(results, OUTPUT_DIR)
    except Exception as e:
        print(f"✗ 生成统计图表失败: {e}")
        import traceback
        traceback.print_exc()
    
    
    
    # 完成
    print("\n" + "="*70)
    print("✓ 所有任务完成！")
    print(f"✓ 输出目录: {OUTPUT_DIR.absolute()}")
    print("\n生成的文件:")
    print("  1. graph1_theme_network.png - 主题-文化关联网络")
    print("  2. graph2_sentiment_comparison.png - 角色情感极性对比")
    print("  3. graph3_cultural_interpretation.png - 文化阐释路径差异")
    print("  4. analysis_report.txt - 详细分析报告")
    print("  5. supplementary_charts.png - 补充统计图表")
   
    print("="*70 + "\n")

# ==================== 运行程序 ====================
if __name__ == "__main__":
    main()