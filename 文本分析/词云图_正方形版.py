"""
进击的巨人跨文化评论词云图生成系统
集成数据加载、处理和词云生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP库
import jieba
from wordcloud import WordCloud

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 文化关键词配置 ====================

class CulturalKeywordManager:
    """文化关键词管理器"""
    
    def __init__(self):
        self.cultural_keywords = self._load_cultural_keywords()
        self.stopwords = self._load_stopwords()
    
    def _load_cultural_keywords(self):
        """加载文化关键词"""
        return {
            'zh': {
                # 核心主题词（最高权重）
                '自由': 120, '自由意志': 120, '解放': 100, '束缚': 100,
                '战争': 120, '和平': 110, '反战': 110, '暴力': 90, '牺牲': 100,
                '种族': 120, '民族': 120, '歧视': 110, '仇恨': 100, '和解': 110,
                '复仇': 120, '报复': 100, '宽恕': 110, '救赎': 110,
                '命运': 120, '宿命': 110, '选择': 110, '轮回': 100, '因果': 100,
                '正义': 120, '邪恶': 100, '善恶': 110, '道德': 110, '立场': 100,
                
                # 角色名（高权重）
                '艾伦': 90, '三笠': 80, '阿尔敏': 80, '利威尔': 80, '兵长': 80,
                '埃尔文': 70, '莱纳': 70, '吉克': 70, '让': 60, '萨莎': 60,
                '希斯特利亚': 70, '韩吉': 60,
                
                # 文化符号词（高权重）
                '巨人': 90, '进击': 90, '调查兵团': 80, '地鸣': 100, 
                '马莱': 80, '艾尔迪亚': 80, '恶魔': 80, '墙内': 70, '墙外': 70,
                '自由之翼': 100, '献出心脏': 90, '始祖巨人': 80, '智慧巨人': 70,
                
                # 情感倾向词（中等权重）
                '震撼': 70, '感动': 70, '深刻': 70, '神作': 70, '烂尾': 70,
                '失望': 70, '可惜': 70, '遗憾': 70, '意难平': 70, '封神': 70,
                '唏嘘': 70, '精彩': 70,
                
                # 文化阐释词（高权重）
                '历史': 100, '隐喻': 100, '现实映射': 100, '人性': 100, '思考': 90,
                '反思': 90, '价值观': 90, '意义': 90, '深度': 80, '映射': 90,
                '罪恶': 90, '集体': 80, '个人': 80, '传统': 80, '现实': 80,
                '伏笔': 80, '叙事': 90, '残酷': 90,
            },
            'en': {
                # Core themes
                'freedom': 120, 'liberty': 120, 'oppression': 110, 'slavery': 100,
                'war': 120, 'peace': 110, 'violence': 100, 'genocide': 110, 'sacrifice': 100,
                'race': 120, 'racism': 120, 'discrimination': 110, 'hatred': 100, 'reconciliation': 110,
                'revenge': 120, 'vengeance': 100, 'forgiveness': 110, 'redemption': 110,
                'fate': 120, 'destiny': 110, 'choice': 110, 'determinism': 100,
                'morality': 120, 'ethics': 110, 'justice': 120, 'evil': 100, 'moral': 100,
                
                # Characters
                'eren': 90, 'mikasa': 80, 'armin': 80, 'levi': 80, 'erwin': 70,
                'reiner': 70, 'zeke': 70, 'jean': 60, 'sasha': 60,
                'historia': 70, 'hange': 60,
                
                # Cultural symbols
                'titan': 90, 'rumbling': 100, 'marley': 80, 'eldian': 80, 'paradis': 80,
                'devil': 80, 'wall': 70, 'wings': 100, 'founder': 80, 'titan shifter': 70,
                
                # Emotional tendency
                'masterpiece': 70, 'brilliant': 70, 'disappointed': 70, 'terrible': 70,
                'amazing': 70, 'awful': 70, 'controversial': 80, 'phenomenal': 70,
                
                # Cultural interpretation
                'holocaust': 100, 'historical': 100, 'parallel': 100, 'humanity': 100,
                'philosophy': 90, 'meaning': 90, 'depth': 90, 'symbolism': 90,
                'individual': 80, 'collective': 80, 'perspective': 90,
                'human rights': 100, 'moral complexity': 90, 'existential': 80,
            },
            'ja': {
                # コアテーマ
                '自由': 120, '自由の翼': 120, '解放': 100, '束縛': 100,
                '戦争': 120, '平和': 110, '暴力': 100, '犠牲': 100, '反戦': 110,
                '民族': 120, '人種': 120, '差別': 110, '迫害': 100, '共存': 110,
                '復讐': 120, '報復': 100, '許し': 110, '贖罪': 110,
                '運命': 120, '宿命': 110, '選択': 110, '未来': 100, '道': 100,
                '正義': 120, '悪': 100, '善悪': 110, '倫理': 110, '立場': 100,
                
                # キャラクター
                'エレン': 90, 'ミカサ': 80, 'アルミン': 80, 'リヴァイ': 80, '兵長': 80,
                'エルヴィン': 70, 'ライナー': 70, 'ジーク': 70, 'ジャン': 60, 'サシャ': 60,
                'ヒストリア': 70, 'ハンジ': 60,
                
                # 文化記号
                '巨人': 90, '進撃': 90, '調査兵団': 80, '地鳴らし': 100,
                'マーレ': 80, 'エルディア': 80, '悪魔': 80, '壁': 70,
                '始祖の巨人': 80, '自由への進撃': 100, '知性巨人': 70,
                
                # 感情傾向
                '神作': 70, '最高': 70, '感動': 70, '残念': 70, '微妙': 70,
                '素晴らしい': 70, 'がっかり': 70, '複雑': 80, '傑作': 70,
                
                # 文化解釈
                '歴史': 100, '現実': 100, '人間性': 100, '哲学': 90, '意味': 90,
                '深い': 90, '考えさせられる': 90, '象徴': 90, '視点': 90,
                '個人': 80, '集団': 80, '伝統': 80, '価値観': 90,
                '戦争の悲劇': 100, '差別の構造': 100, '未来への希望': 90,
            }
        }
    
    def _load_stopwords(self):
        """加载停用词"""
        return {
            'zh': {
                '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', 
                '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
                '会', '着', '没有', '看', '好', '自己', '这', '能', '那', '里',
                '后', '以', '这个', '为', '来', '而', '个', '地', '可', '这样',
                '然后', '但', '但是', '所以', '因为', '如果', '还', '就是', '啊',
                '吗', '呢', '吧', '哦', '嗯', '哈', '哈哈', '我们', '他', '她',
                '它', '他们', '这些', '那些', '什么', '怎么', '为什么', '多少',
                '太', '真', '觉得', '感觉', '真的', '实在', '非常', '特别',
                '动漫', '作品', '番剧', '动画', '漫画', '剧情', '故事', '情节',
                '集', '季', '最后', '结局', '开始', '最终', '进击的巨人', '进击',
                '看完', '看到', '看了', '观众', '评论', '豆瓣', '评分', '这部',
                '这部作品', '这部动漫', '这部动画', '这个', '这部漫画',
            },
            'en': {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'them',
                'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
                'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'just', 'about', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'between',
                'episode', 'season', 'watch', 'watched', 'watching', 'show', 'anime',
                's', 't', 've', 'll', 're', 'd', 'don', 'didn', 'doesn', 'isn', 'wasn',
                'attack', 'on', 'titan', 'aot', 'snk', 'series', 'story', 'plot',
                'final', 'ending', 'end', 'start', 'finished', 'scene', 'character',
                'think', 'feel', 'really', 'get', 'got', 'see', 'saw', 'make', 'made',
                'this anime', 'this show', 'this series', 'this story',
            },
            'ja': {
                'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ',
                'さ', 'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として',
                'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その',
                'あっ', 'よう', 'また', 'もの', 'という', 'あり', 'まで', 'られ',
                'なる', 'へ', 'か', 'だ', 'これ', 'によって', 'により', 'おり',
                'より', 'による', 'ず', 'なり', 'られる', 'において', 'ば', 'なかっ',
                'なく', 'しかし', 'について', 'せ', 'だっ', 'その後', 'できる', 'それ',
                'う', 'ので', 'なお', 'のみ', 'でき', 'き', 'つ', 'における',
                'および', 'いう', 'さらに', 'でも', 'ら', 'たり', 'その他', 'に関する',
                'たち', 'ます', 'ん', 'なら', 'に対して', '特に', 'せる', '及び',
                'これら', 'とき', 'では', 'にて', 'ほか', 'ながら', 'うち', 'そして',
                'とともに', 'ただし', 'かつて', 'それぞれ', 'または', 'お', 'ほど',
                'ものの', 'に対する', 'ほとんど', 'と共に', 'といった', 'です', 'あります',
                'でした', 'ました', 'します', 'アニメ', '作品', '話', '最終', '回', '见',
                '進撃の巨人', '物語', 'ストーリー', 'シーズン', '最後', '結末', 'この',
                '思う', '感じる', '本当', '見た', '見る', 'キャラクター', 'キャラ',
                'このアニメ', 'この作品', 'この物語', 'このストーリー',
            }
        }


# ==================== 数据加载模块 ====================

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.dataframes = {}
    
    def load_data(self, file_paths):
        """
        加载所有数据
        
        参数:
        file_paths: dict, 格式为 {'zh': [路径列表], 'en': [路径列表], 'ja': [路径列表]}
        """
        print("="*60)
        print("开始加载评论数据...")
        print("="*60)
        
        for lang, paths in file_paths.items():
            print(f"\n【加载{lang}数据】")
            
            all_data = []
            for i, path in enumerate(paths, 1):
                try:
                    print(f"  文件 {i}/{len(paths)}: {Path(path).name}")
                    df = self._load_single_file(path, lang)
                    if df is not None and not df.empty:
                        all_data.append(df)
                        print(f"    ✓ 加载成功: {len(df)} 条")
                    else:
                        print(f"    ✗ 加载失败或文件为空")
                except Exception as e:
                    print(f"    ✗ 加载失败: {str(e)}")
            
            if all_data:
                # 合并数据
                combined = pd.concat(all_data, ignore_index=True)
                # 清理和去重
                combined = self._clean_dataframe(combined, lang)
                self.dataframes[lang] = combined
                print(f"  ✓ {lang}数据准备完成: {len(combined)} 条")
            else:
                print(f"  ✗ 没有有效的{lang}数据")
        
        return self.dataframes
    
    def _load_single_file(self, file_path, lang):
        """加载单个文件"""
        path = Path(file_path)
        if not path.exists():
            return None
        
        try:
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    break
                except:
                    continue
            else:
                return None
            
            # 重命名列
            df = self._standardize_columns(df, lang)
            return df
            
        except Exception as e:
            print(f"    读取错误: {e}")
            return None
    
    def _standardize_columns(self, df, lang):
        """标准化列名"""
        df = df.copy()
        
        # 可能的列名映射
        column_maps = {
            'zh': {
                'text': ['comment', '评论', 'content', '内容', 'text', '评论内容'],
                'rating': ['rating', '评分', 'score', 'stars'],
                'votes': ['votes', '有用', '点赞', 'like']
            },
            'en': {
                'text': ['comment', 'review', 'text', 'content', 'review_text'],
                'rating': ['rating', 'score', 'overall', 'stars'],
                'votes': ['votes', 'helpful', 'likes', 'upvotes']
            },
            'ja': {
                'text': ['comment', '评论', 'レビュー', '内容', 'text'],
                'rating': ['rating', '评分', '評価', 'score'],
                'votes': ['votes', 'いいね', '役に立った', 'like']
            }
        }
        
        # 重命名列
        for target_col, possible_names in column_maps[lang].items():
            for name in possible_names:
                if name in df.columns:
                    df = df.rename(columns={name: target_col})
                    break
        
        # 确保有text列
        if 'text' not in df.columns:
            # 查找可能是文本的列
            for col in df.columns:
                if df[col].dtype == 'object' and len(df) > 0:
                    sample = str(df[col].iloc[0])
                    if len(sample) > 10:  # 认为是评论
                        df = df.rename(columns={col: 'text'})
                        break
            else:
                # 如果没有找到，创建空的text列
                df['text'] = ''
        
        return df
    
    def _clean_dataframe(self, df, lang):
        """清理数据框"""
        if df.empty:
            return df
        
        # 确保text是字符串
        df['text'] = df['text'].astype(str).str.strip()
        
        # 过滤空评论和短评论
        df = df[df['text'].str.len() > 10]
        
        # 去重
        df = df.drop_duplicates(subset=['text'])
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df


# ==================== 词云生成模块 ====================

class SquareWordCloudGenerator:
    """正方形词云生成器"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keyword_manager = CulturalKeywordManager()
    
    def generate_all(self, dataframes):
        """生成所有词云"""
        print("\n" + "="*60)
        print("开始生成词云图...")
        print("="*60)
        
        results = []
        
        for lang, df in dataframes.items():
            if lang not in ['zh', 'en', 'ja'] or df.empty:
                continue
            
            print(f"\n【生成{lang}词云】")
            
            try:
                # 提取文本
                texts = ' '.join(df['text'].astype(str).tolist())
                
                if len(texts) < 100:
                    print(f"  ✗ 文本太少: {len(texts)} 字符")
                    continue
                
                # 处理文本并提取关键词
                word_freq = self._process_text(texts, lang)
                
                if not word_freq:
                    print(f"  ✗ 未提取到关键词")
                    continue
                
                print(f"  关键词数量: {len(word_freq)}")
                print(f"  高频词: {list(word_freq.keys())[:10]}")
                
                # 生成词云
                output_file = self._generate_wordcloud(word_freq, lang)
                if output_file:
                    results.append(output_file)
                    print(f"  ✓ 已保存: {output_file.name}")
                    
            except Exception as e:
                print(f"  ✗ 生成失败: {str(e)}")
        
        return results
    
    def _process_text(self, text, lang):
        """处理文本并提取关键词"""
        # 分词
        if lang in ['zh', 'ja']:
            words = list(jieba.cut(text))
        else:
            words = text.lower().split()
            words = [''.join(c for c in w if c.isalnum() or c == '-') for w in words]
        
        # 统计词频
        word_freq = Counter(words)
        
        # 获取关键词和停用词
        cultural_keywords = self.keyword_manager.cultural_keywords[lang]
        stopwords = self.keyword_manager.stopwords[lang]
        
        # 计算加权频率
        weighted_freq = {}
        for word, count in word_freq.items():
            # 过滤条件
            if (len(word) < 2 or 
                word in stopwords or 
                word.isdigit() or
                self._is_punctuation(word)):
                continue
            
            # 基础得分
            base_score = np.log1p(count) * 10
            
            # 文化权重加成
            cultural_weight = cultural_keywords.get(word, 0)
            
            if cultural_weight > 0:
                if cultural_weight >= 100:  # 核心词
                    final_score = base_score * (1 + cultural_weight / 50)
                elif cultural_weight >= 80:  # 重要词
                    final_score = base_score * (1 + cultural_weight / 70)
                else:  # 一般词
                    final_score = base_score * (1 + cultural_weight / 100)
            else:
                # 非文化词降权
                final_score = base_score * 0.05
            
            weighted_freq[word] = int(final_score)
        
        # 确保核心关键词出现
        for keyword, weight in cultural_keywords.items():
            if weight >= 100 and keyword in text:
                if keyword not in weighted_freq or weighted_freq[keyword] < weight:
                    weighted_freq[keyword] = weight
        
        # 取前100个
        sorted_words = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:100]
        
        return dict(sorted_words)
    
    def _is_punctuation(self, text):
        """判断是否为标点"""
        punctuation = set('!?。，、；：""''（）【】《》~@#￥%……&*·—\n\r\t ,./;\'[]\\<>?:"{}|!@#$%^&*()_+-=~`')
        return all(c in punctuation for c in text)
    
    def _get_font_path(self, lang):
        """获取字体路径"""
        import platform
        system = platform.system()
        
        font_paths = {
            'zh': {
                'Windows': ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc'],
                'Darwin': ['/System/Library/Fonts/PingFang.ttc'],
                'Linux': ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']
            },
            'ja': {
                'Windows': ['C:/Windows/Fonts/meiryo.ttc', 'C:/Windows/Fonts/msgothic.ttc'],
                'Darwin': ['/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'],
            }
        }
        
        if lang in font_paths:
            paths = font_paths[lang].get(system, [])
            for path in paths:
                if Path(path).exists():
                    return path
        
        return None
    
    def _generate_wordcloud(self, word_freq, lang):
        """生成词云图像"""
        # 配色方案
        color_schemes = {
            'zh': {
                'bg_color': '#0D0D0D',
                'colors': ['#C62828', '#B71C1C', '#D32F2F', '#E53935', '#F44336',
                          '#FF8F00', '#FF9800', '#FFA000', '#FFB300', '#FFC107'],
                'title': '进击的巨人 - 中国观众文化阐释',
                'title_color': '#FFD700'
            },
            'en': {
                'bg_color': '#1A237E',
                'colors': ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5',
                          '#78909C', '#90A4AE', '#B0BEC5', '#CFD8DC', '#ECEFF1'],
                'title': 'Attack on Titan - Western Cultural Interpretation',
                'title_color': '#42A5F5'
            },
            'ja': {
                'bg_color': '#311B92',
                'colors': ['#E91E63', '#D81B60', '#C2185B', '#AD1457',
                          '#9C27B0', '#8E24AA', '#7B1FA2', '#6A1B9A'],
                'title': '進撃の巨人 - 日本文化解釈',
                'title_color': '#E91E63'
            }
        }
        
        scheme = color_schemes[lang]
        
        # 自定义颜色函数
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return np.random.choice(scheme['colors'])
        
        # 获取字体
        font_path = self._get_font_path(lang)
        
        # 创建词云
        wc = WordCloud(
            font_path=font_path,
            width=1600,
            height=1600,
            background_color=scheme['bg_color'],
            max_words=100,
            relative_scaling=0.4,
            min_font_size=12,
            max_font_size=180,
            contour_width=2,
            contour_color=scheme['colors'][0],
            stopwords=set(),
            collocations=False,
            prefer_horizontal=0.7,
            margin=5,
            color_func=color_func,
            random_state=42
        )
        
        wc.generate_from_frequencies(word_freq)
        
        # 保存图像
        plt.figure(figsize=(12, 12), facecolor=scheme['bg_color'])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(scheme['title'], fontsize=20, fontweight='bold', 
                 color=scheme['title_color'], pad=20)
        
        output_file = self.output_dir / f'wordcloud_{lang}_square.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                   facecolor=scheme['bg_color'], edgecolor='none')
        plt.close()
        
        return output_file


# ==================== 主程序 ====================

def main():
    """主程序"""
    print("="*70)
    print("进击的巨人跨文化评论词云图生成系统")
    print("单文件完整版")
    print("="*70)
    
    # 配置文件路径
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
    
    # 输出目录
    output_dir = r'D:\用户数据采集\词云图输出\正方形词云'
    
    try:
        # 1. 加载数据
        print("\n阶段1: 数据加载")
        print("-"*40)
        
        loader = DataLoader()
        dataframes = loader.load_data(file_paths)
        
        if not dataframes:
            print("\n❌ 没有加载到任何数据！")
            return
        
        # 显示统计数据
        print("\n数据统计:")
        for lang, df in dataframes.items():
            print(f"  {lang}: {len(df)} 条评论")
        
        total = sum(len(df) for df in dataframes.values())
        print(f"总计: {total} 条评论")
        
        # 2. 生成词云
        print("\n阶段2: 生成词云图")
        print("-"*40)
        
        generator = SquareWordCloudGenerator(output_dir)
        results = generator.generate_all(dataframes)
        
        # 3. 显示结果
        if results:
            print("\n" + "="*70)
            print("✅ 词云生成完成！")
            print("\n生成的文件:")
            for file in results:
                print(f"  - {file.name}")
            print(f"\n保存路径: {output_dir}")
            print("="*70)
        else:
            print("\n❌ 词云生成失败！")
        
        # 4. 保存处理后的数据（可选）
        print("\n阶段3: 保存处理后的数据")
        print("-"*40)
        
        processed_dir = Path(output_dir) / 'processed_data'
        processed_dir.mkdir(exist_ok=True)
        
        for lang, df in dataframes.items():
            save_path = processed_dir / f'processed_{lang}.csv'
            # 只保存text列
            df[['text']].to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存{lang}数据: {save_path.name}")
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试"""
    print("快速测试模式...")
    
    # 使用部分文件测试
    test_paths = {
        'zh': [
            r'D:\用户数据采集\数据爬取\豆瓣\douban_27072327_comments_latest.csv'
        ],
        'en': [
            r'D:\用户数据采集\数据爬取\myanimelist\mal_data\AOT_ALL_SEASONS_20251128_170141.csv'
        ]
    }
    
    output_dir = r'D:\用户数据采集\词云图输出\快速测试'
    
    try:
        # 加载数据
        loader = DataLoader()
        dataframes = loader.load_data(test_paths)
        
        if dataframes:
            # 生成词云
            generator = SquareWordCloudGenerator(output_dir)
            results = generator.generate_all(dataframes)
            
            if results:
                print(f"\n✅ 测试成功！生成 {len(results)} 个词云")
                for file in results:
                    print(f"  - {file.name}")
            else:
                print("\n❌ 测试失败！")
        else:
            print("❌ 没有加载到数据！")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")


# ==================== 运行程序 ====================

if __name__ == "__main__":
    print("进击的巨人词云图生成系统")
    print("1. 完整运行")
    print("2. 快速测试")
    
    try:
        choice = input("请选择 (1或2, 默认1): ").strip()

    except:
        choice = '1'
    
    if choice == '2':
        quick_test()
    else:
        main()
    
    print("\n程序执行完毕！")
    input("按Enter键退出...")