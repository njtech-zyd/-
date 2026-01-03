import requests
from bs4 import BeautifulSoup
import time
import csv
import random
import os

class DoubanCommentSpider:
    def __init__(self, movie_id, cookie_str):
        self.movie_id = movie_id
        self.base_url = f'https://movie.douban.com/subject/{movie_id}/comments'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': f'https://movie.douban.com/subject/{movie_id}/',
            'Cookie': cookie_str
        }
        # 【修改点1】文件名加上 _latest，表示按时间排序，避免覆盖刚才的文件
        self.csv_filename = f'douban_{movie_id}_comments_latest.csv'
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', encoding='utf-8-sig', newline='') as f:
                fieldnames = ['username', 'rating', 'time', 'votes', 'comment']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def save_page_to_csv(self, comments):
        """增量保存"""
        if not comments:
            return
        with open(self.csv_filename, 'a', encoding='utf-8-sig', newline='') as f:
            fieldnames = ['username', 'rating', 'time', 'votes', 'comment']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(comments)

    def get_page_comments(self, start=0):
        """获取单页评论"""
        params = {
            'start': start,
            'limit': 20,
            'status': 'P',
            # 【修改点2】这里改成了 'time'，获取最新评论
            'sort': 'time' 
        }
        
        try:
            response = requests.get(
                self.base_url, 
                headers=self.headers, 
                params=params,
                timeout=15
            )
            
            if response.status_code == 403:
                print(f"!!! 警告：403 Forbidden。IP被封或Cookie过期。")
                return [], False
                
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            comment_items = soup.find_all('div', class_='comment-item')
            if not comment_items:
                return [], False
            
            page_comments = []
            for item in comment_items:
                comment_data = {}
                
                user = item.find('span', class_='comment-info')
                if user and user.find('a'):
                    comment_data['username'] = user.find('a').text.strip()
                
                rating = item.find('span', class_='rating')
                if rating:
                    rating_class = rating.get('class')
                    stars = [x for x in rating_class if 'allstar' in x]
                    if stars:
                        comment_data['rating'] = int(stars[0].replace('allstar', '')) // 10
                    else:
                        comment_data['rating'] = '未评分'
                else:
                    comment_data['rating'] = '未评分'
                
                comment_time = item.find('span', class_='comment-time')
                if comment_time:
                    comment_data['time'] = comment_time.text.strip().replace('\n', '').strip()
                
                comment_text = item.find('span', class_='short')
                if comment_text:
                    comment_data['comment'] = comment_text.text.strip()
                
                votes = item.find('span', class_='votes')
                if votes:
                    comment_data['votes'] = votes.text.strip()
                
                page_comments.append(comment_data)
            
            next_btn = soup.find('a', class_='next')
            has_next = True if next_btn else False
            
            return page_comments, has_next
            
        except Exception as e:
            print(f"请求出错 (start={start}): {e}")
            return [], False
    
    def crawl_all(self):
        """主循环"""
        start = 0
        page_num = 1
        total_count = 0
        
        print(f"=== 启动爬取 (按最新时间排序) ===")
        print(f"目标电影ID: {self.movie_id}")
        print(f"结果将保存至: {self.csv_filename}")
        
        while True:
            print(f"正在获取第 {page_num} 页 (start={start})...")
            
            comments, has_next = self.get_page_comments(start)
            
            if not comments:
                print("未获取到数据，停止爬取。")
                break
            
            self.save_page_to_csv(comments)
            count = len(comments)
            total_count += count
            print(f"  -> 成功保存 {count} 条 (累计: {total_count})")
            
            if not has_next:
                print("已到达最后一页。")
                break
            
            start += 20
            page_num += 1
            
            # 随机延时
            sleep_time = random.uniform(3, 6)
            time.sleep(sleep_time)
            
        print("-" * 30)
        print(f"爬取完成！共获取 {total_count} 条最新评论。")

if __name__ == '__main__':
    # 【主要修改点】将电影ID改为 27072327
    target_movie_id = '27072327'
    
    # 你的 Cookie (保持不变)
    my_cookie = input("请粘贴你的豆瓣Cookie: ").strip()
    spider = DoubanCommentSpider(target_movie_id, my_cookie)
    spider.crawl_all()