"""
MyAnimeList 多季度爬虫 - 增强版
支持断点续传、自动重试、实时保存
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime
import re
import os
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RobustMALScraper:
    AOT_SEASONS = {
        'Season_3_Part_1': 'https://myanimelist.net/anime/35760/Shingeki_no_Kyojin_Season_3/reviews',
        'Season_3_Part_2': 'https://myanimelist.net/anime/38524/Shingeki_no_Kyojin_Season_3_Part_2/reviews',
    }
    
    def __init__(self, output_dir='mal_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        self.all_reviews = []
        self.progress_file = os.path.join(output_dir, 'progress.json')
        self.progress = self.load_progress()
    
    def create_session(self):
        """创建带重试机制的session"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=5,  # 总共重试5次
            backoff_factor=2,  # 指数退避
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        
        return session
    
    def load_progress(self):
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_progress(self, season, page):
        """保存进度"""
        self.progress[season] = page
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f)
    
    def parse_review(self, review_div, season_name):
        """解析评论"""
        try:
            data = {}
            
            # 用户名
            username_elem = review_div.find('div', class_='username')
            if username_elem:
                user_link = username_elem.find('a')
                data['username'] = user_link.text.strip() if user_link else 'Unknown'
            else:
                data['username'] = 'Unknown'
            
            # 评论文本
            text_elem = review_div.find('div', class_='text')
            if text_elem:
                for unwanted in text_elem.find_all(['a', 'div']):
                    unwanted.decompose()
                data['comment'] = text_elem.get_text(separator=' ', strip=True)
            else:
                data['comment'] = review_div.get_text(separator=' ', strip=True)
            
            if len(data['comment']) < 20:
                return None
            
            # 评分
            all_text = review_div.get_text()
            
            overall_elem = review_div.find('div', class_='rating')
            if overall_elem:
                match = re.search(r'(\d+)', overall_elem.get_text(strip=True))
                data['rating'] = match.group(1) if match else 'N/A'
            else:
                data['rating'] = 'N/A'
            
            # 日期
            date_elem = review_div.find('div', class_='date')
            if date_elem:
                data['time'] = date_elem.get_text(strip=True)
            else:
                date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+,\s+\d{4}', all_text)
                data['time'] = date_match.group(0) if date_match else 'N/A'
            
            # 有用度
            helpful_match = re.search(r'(\d+)\s+of\s+(\d+)', all_text)
            if helpful_match:
                data['votes'] = f"{helpful_match.group(1)}/{helpful_match.group(2)}"
            else:
                data['votes'] = '0/0'
            
            # 元数据
            data['season'] = season_name
            data['scraped_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return data
            
        except Exception as e:
            return None
    
    def scrape_page_with_retry(self, url, page_num, season_name, max_retries=3):
        """爬取单页（带重试）"""
        for attempt in range(max_retries):
            try:
                session = self.create_session()
                response = session.get(url, timeout=30)
                
                if response.status_code != 200:
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return 0
                
                soup = BeautifulSoup(response.text, 'html.parser')
                review_elements = soup.find_all('div', class_='review-element')
                
                if not review_elements:
                    return 0
                
                page_reviews = []
                for elem in review_elements:
                    review_data = self.parse_review(elem, season_name)
                    if review_data:
                        page_reviews.append(review_data)
                
                return page_reviews
                
            except (requests.exceptions.SSLError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                print(f"      ⚠️ 尝试 {attempt+1}/{max_retries}: {type(e).__name__}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"      ⏸️  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"      ❌ 第{page_num}页失败，跳过")
                    return 0
            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断，保存当前数据...")
                self.save_current_data()
                raise
            except Exception as e:
                print(f"      ❌ 未知错误: {e}")
                return 0
        
        return 0
    
    def scrape_season(self, season_name, url, max_pages=30):
        """爬取单个季度"""
        print(f"\n{'='*70}")
        print(f"🎬 开始爬取: {season_name}")
        print(f"{'='*70}")
        
        # 检查是否有断点
        start_page = self.progress.get(season_name, 1)
        if start_page > 1:
            print(f"📍 从第 {start_page} 页继续...")
        
        season_reviews = []
        consecutive_empty = 0
        
        for page in range(start_page, max_pages + 1):
            page_url = url if page == 1 else f"{url}?p={page}"
            
            print(f"📄 第{page}页...", end=' ')
            
            page_reviews = self.scrape_page_with_retry(page_url, page, season_name)
            
            if isinstance(page_reviews, list) and len(page_reviews) > 0:
                season_reviews.extend(page_reviews)
                print(f"✅ {len(page_reviews)}条 (累计: {len(season_reviews)})")
                consecutive_empty = 0
                
                # 保存进度
                self.save_progress(season_name, page)
                
                # 每10页保存一次数据
                if page % 10 == 0:
                    self.save_season_data(season_name, season_reviews)
            else:
                print(f"❌ 0条")
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print(f"      🛑 连续 {consecutive_empty} 页无数据，停止此季")
                    break
            
            # 延迟
            time.sleep(random.uniform(2, 5))
        
        # 保存最终数据
        self.save_season_data(season_name, season_reviews)
        
        print(f"✨ {season_name} 完成: {len(season_reviews)}条评论\n")
        return season_reviews
    
    def save_season_data(self, season_name, reviews):
        """保存单季数据"""
        if not reviews:
            return
        
        # 只保留需要的字段
        filtered_reviews = []
        for review in reviews:
            filtered_review = {
                'username': review.get('username', 'Unknown'),
                'rating': review.get('rating', 'N/A'),
                'time': review.get('time', 'N/A'),
                'votes': review.get('votes', '0/0'),
                'comment': review.get('comment', ''),
                'season': review.get('season', '')
            }
            filtered_reviews.append(filtered_review)
        
        df = pd.DataFrame(filtered_reviews)
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = os.path.join(self.output_dir, f'{season_name}_{timestamp}.csv')
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"      💾 已保存: {filename}")
    
    def save_current_data(self):
        """保存当前所有数据"""
        if not self.all_reviews:
            return
        
        # 只保留需要的字段
        filtered_reviews = []
        for review in self.all_reviews:
            filtered_review = {
                'username': review.get('username', 'Unknown'),
                'rating': review.get('rating', 'N/A'),
                'time': review.get('time', 'N/A'),
                'votes': review.get('votes', '0/0'),
                'comment': review.get('comment', ''),
                'season': review.get('season', '')
            }
            filtered_reviews.append(filtered_review)
        
        df = pd.DataFrame(filtered_reviews)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'AOT_PARTIAL_{timestamp}.csv')
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 已保存当前数据: {filename}")
        print(f"📊 共 {len(df)} 条评论")
    
    def scrape_all_seasons(self):
        """爬取所有季度"""
        print("\n" + "="*70)
        print("🚀 MyAnimeList 进击的巨人全系列爬虫 (增强版)")
        print("="*70)
        print(f"📺 目标季度: {len(self.AOT_SEASONS)}个")
        print(f"💾 数据保存目录: {self.output_dir}")
        print("🔄 支持断点续传和自动重试")
        print("="*70)
        
        start_time = time.time()
        
        try:
            for i, (season_name, url) in enumerate(self.AOT_SEASONS.items(), 1):
                print(f"\n[{i}/{len(self.AOT_SEASONS)}] {season_name}")
                
                season_reviews = self.scrape_season(season_name, url)
                self.all_reviews.extend(season_reviews)
                
                # 季之间延迟
                if i < len(self.AOT_SEASONS):
                    wait = random.uniform(10, 20)
                    print(f"⏸️  等待 {wait:.1f} 秒后继续下一季...\n")
                    time.sleep(wait)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ 爬取被中断")
            self.save_current_data()
            return self.all_reviews
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"🎉 全部完成！")
        print(f"📊 总评论数: {len(self.all_reviews)}")
        print(f"⏱️  总耗时: {elapsed/60:.1f}分钟")
        print("="*70)
        
        return self.all_reviews
    
    def save_final_data(self):
        """保存最终数据"""
        if not self.all_reviews:
            print("⚠️ 没有数据")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 只保留需要的字段
        filtered_reviews = []
        for review in self.all_reviews:
            filtered_review = {
                'username': review.get('username', 'Unknown'),
                'rating': review.get('rating', 'N/A'),
                'time': review.get('time', 'N/A'),
                'votes': review.get('votes', '0/0'),
                'comment': review.get('comment', ''),
                'season': review.get('season', '')
            }
            filtered_reviews.append(filtered_review)
        
        df = pd.DataFrame(filtered_reviews)
        
        # CSV
        csv_file = os.path.join(self.output_dir, f'AOT_ALL_SEASONS_{timestamp}.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 总数据CSV: {csv_file}")
        
        # Excel
        excel_file = os.path.join(self.output_dir, f'AOT_ALL_SEASONS_{timestamp}.xlsx')
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"💾 总数据Excel: {excel_file}")
        
        # 统计报告
        self.generate_report(df, timestamp)
        
        # 清除进度文件
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            print(f"🗑️  已清除进度文件")
        
        return df
    
    def generate_report(self, df, timestamp):
        """生成报告"""
        report_file = os.path.join(self.output_dir, f'REPORT_{timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("进击的巨人 MyAnimeList 评论数据统计报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"【总体统计】\n")
            f.write(f"总评论数: {len(df):,}\n")
            f.write(f"涵盖季度: {df['season'].nunique()}\n")
            f.write(f"独立用户: {df['username'].nunique():,}\n")
            f.write(f"平均评论长度: {df['comment'].str.len().mean():.0f} 字符\n\n")
            
            f.write(f"【各季度评论数】\n")
            for season, count in df['season'].value_counts().items():
                f.write(f"{season:25s}: {count:4d}\n")
            f.write("\n")
            
            f.write(f"【评分分布】\n")
            ratings = df[df['rating'] != 'N/A']['rating'].value_counts().sort_index()
            for rating, count in ratings.items():
                f.write(f"{rating}分: {count:4d}\n")
        
        print(f"📊 统计报告: {report_file}")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      MyAnimeList 进击的巨人全系列评论爬虫 - 增强版          ║
    ║      支持断点续传 | 自动重试 | 实时保存                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n💡 新功能:")
    print("  ✅ 自动重试失败的页面")
    print("  ✅ 遇到中断可以继续（保存进度）")
    print("  ✅ 每10页自动保存数据")
    print("  ✅ 增强的SSL错误处理")
    
    choice = input("\n选择模式:\n1. 爬取所有季度\n2. 只爬取指定季度\n请输入(1/2): ").strip()
    
    scraper = RobustMALScraper()
    
    if choice == '2':
        print("\n可用季度:")
        seasons = list(scraper.AOT_SEASONS.keys())
        for i, s in enumerate(seasons, 1):
            print(f"{i}. {s}")
        
        selected = input("\n输入编号(逗号分隔): ").strip()
        indices = [int(x.strip())-1 for x in selected.split(',') if x.strip().isdigit()]
        
        selected_seasons = {seasons[i]: scraper.AOT_SEASONS[seasons[i]] 
                          for i in indices if 0 <= i < len(seasons)}
        scraper.AOT_SEASONS = selected_seasons
    
    print("\n" + "="*70)
    print(f"将爬取 {len(scraper.AOT_SEASONS)} 个季度")
    input("\n按 Enter 开始...")
    
    try:
        # 开始爬取
        reviews = scraper.scrape_all_seasons()
        
        # 保存最终数据
        if reviews:
            df = scraper.save_final_data()
            
            print(f"\n✅ 完成！")
            print(f"📁 所有数据保存在: {scraper.output_dir}")
            print(f"📊 总计: {len(df)} 条评论")
    
    except KeyboardInterrupt:
        print("\n\n👋 程序已终止")
        print(f"💾 已保存的数据在: {scraper.output_dir}")
        print(f"💡 下次运行会从中断处继续")


if __name__ == "__main__":
    main()