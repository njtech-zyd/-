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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'https://movie.douban.com/subject/{movie_id}/',
            'Cookie': cookie_str,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        self.csv_filename = f'douban_{movie_id}_comments.csv'
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', encoding='utf-8-sig', newline='') as f:
                fieldnames = ['username', 'rating', 'time', 'votes', 'comment']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def save_page_to_csv(self, comments):
        """把当前页的数据追加写入CSV"""
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
            'sort': 'new_score'
        }
        
        try:
            response = requests.get(
                self.base_url, 
                headers=self.headers, 
                params=params,
                timeout=15
            )
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 403:
                print("!!! 403 Forbidden - 可能需要更新Cookie")
                return [], False
            elif response.status_code == 404:
                print("!!! 404 Not Found - 电影ID可能错误")
                return [], False
            elif response.status_code == 401:
                print("!!! 401 Unauthorized - Cookie无效或已过期")
                return [], False
                
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 检查反爬提示
            if "检测到有异常请求" in response.text:
                print("!!! 被反爬虫机制检测到")
                return [], False
                
            comment_items = soup.find_all('div', class_='comment-item')
            print(f"找到 {len(comment_items)} 条评论")
            
            if not comment_items:
                return [], False
            
            page_comments = []
            for item in comment_items:
                comment_data = {}
                
                # 用户名
                user = item.find('span', class_='comment-info')
                if user and user.find('a'):
                    comment_data['username'] = user.find('a').text.strip()
                else:
                    comment_data['username'] = '未知用户'
                
                # 评分
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
                
                # 时间
                comment_time = item.find('span', class_='comment-time')
                if comment_time:
                    comment_data['time'] = comment_time.text.strip()
                else:
                    comment_data['time'] = '未知时间'
                
                # 评论内容
                comment_text = item.find('span', class_='short')
                if comment_text:
                    comment_data['comment'] = comment_text.text.strip()
                else:
                    comment_data['comment'] = '无内容'
                
                # 点赞数
                votes = item.find('span', class_='votes')
                if votes:
                    comment_data['votes'] = votes.text.strip()
                else:
                    comment_data['votes'] = '0'
                
                page_comments.append(comment_data)
            
            # 检查是否有下一页
            next_btn = soup.find('a', class_='next')
            has_next = True if next_btn else False
            
            return page_comments, has_next
            
        except Exception as e:
            print(f"请求出错: {e}")
            return [], False
    
    def crawl_all(self):
        """主循环：自动翻页直到没有下一页"""
        start = 0
        page_num = 1
        total_count = 0
        
        print(f"目标电影ID: {self.movie_id}")
        print(f"保存文件: {self.csv_filename}")
        print("-" * 40)
        
        while True:
            print(f"第 {page_num} 页 (start={start})...")
            
            comments, has_next = self.get_page_comments(start)
            
            if not comments:
                if page_num == 1:
                    print("!!! 第一页就失败，请检查：")
                    print("1. Cookie是否有效（重要！）")
                    print("2. 电影ID是否正确")
                    print("3. 网络连接")
                break
            
            self.save_page_to_csv(comments)
            count = len(comments)
            total_count += count
            print(f"  → 保存 {count} 条 (累计: {total_count})")
            
            if not has_next:
                print("已到最后一页")
                break
            
            start += 20
            page_num += 1
            
            # 延时
            sleep_time = random.uniform(3, 6)
            time.sleep(sleep_time)
            
        print("-" * 40)
        print(f"完成！共获取 {total_count} 条评论")

def get_cookie_instructions():
    """显示获取Cookie的说明"""
    print("=" * 60)
    print("如何获取豆瓣Cookie：")
    print("1. 在浏览器中登录豆瓣网站 (https://www.douban.com)")
    print("2. 按F12打开开发者工具")
    print("3. 点击Network标签")
    print("4. 刷新页面")
    print("5. 点击任意请求，在Headers中找到Cookie")
    print("6. 复制Cookie的全部内容")
    print("=" * 60)

if __name__ == '__main__':
    # 显示获取Cookie的说明
    get_cookie_instructions()
    
    # 电影ID
    target_movie_id = '36283000'  # 这里可以修改为你想爬取的电影ID 比如30353357，27072327

    
    # 在这里粘贴你获取的新Cookie
    # 注意：Cookie需要经常更新，特别是登录相关的字段
    my_cookie = input("请粘贴你的豆瓣Cookie: ").strip()
    
    if not my_cookie:
        print("未提供Cookie，使用示例Cookie（可能无效）")
        my_cookie = "ll=\"118159\"; bid=example; dbcl2=\"user:example\"; ck=example"
    
    # 启动爬虫
    spider = DoubanCommentSpider(target_movie_id, my_cookie)
    spider.crawl_all()