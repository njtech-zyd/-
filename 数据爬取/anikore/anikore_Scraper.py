import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

def scrape_anikore_reviews(anime_id, target_count=1000):
    """
    根据用户提供的HTML结构爬取anikore.jp评论
    """
    base_url = f"https://www.anikore.jp/anime_review/{anime_id}/"
    
    reviews_data = []
    page = 1
    
    # 伪装请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Referer': 'https://www.anikore.jp/'
    }

    print(f"开始爬取，目标数量: {target_count} 条...")

    while len(reviews_data) < target_count:
        # 构造分页 URL
        url = base_url if page == 1 else f"{base_url}page:{page}"
        
        print(f"正在抓取第 {page} 页: {url} | 当前已收集: {len(reviews_data)} 条")

        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"页面请求失败，状态码: {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 定位评论容器
            # 根据Anikore的结构，m-reviewUnit... 这些元素通常被包裹在 l-animeDetailReview__unit 中
            # 我们查找包含这些特定子元素的容器
            review_blocks = soup.select(".l-animeDetailReview__unit")
            
            # 如果上一级选择器找不到，尝试直接找包含特定类的父级（备用方案）
            if not review_blocks:
                 review_blocks = soup.select(".m-reviewUnit")

            if not review_blocks:
                print("未找到评论块，可能已到达最后一页或页面结构再次变更。")
                break

            for block in review_blocks:
                try:
                    # --- 1. 用户名 (Username) ---
                    # 结构: <p class="m-reviewUnit_userText_nickname"><strong>りは</strong>...
                    user_tag = block.select_one(".m-reviewUnit_userText_nickname strong")
                    username = user_tag.text.strip() if user_tag else "Unknown"

                    # --- 2. 评分 (Rating) ---
                    # 结构: <p class="m-reviewUnit_userText_pointLane">...<strong>4.9</strong>...
                    rating_tag = block.select_one(".m-reviewUnit_userText_pointLane strong")
                    rating = rating_tag.text.strip() if rating_tag else "0"

                    # --- 3. 评论内容 (Comment) ---
                    # 结构: <p class="m-reviewUnit_userText_content ateval_description">
                    content_tag = block.select_one(".m-reviewUnit_userText_content")
                    if content_tag:
                        # separator="\n" 可以把 <br> 标签转换为换行符，保持评论格式
                        comment = content_tag.get_text(separator="\n").strip()
                    else:
                        comment = ""

                    # --- 4. 点赞数 (Votes) ---
                    # 结构: <div class="m-reviewUnit_userText_footerLane_thanks"><p>0</p>
                    votes_tag = block.select_one(".m-reviewUnit_userText_footerLane_thanks p")
                    votes = votes_tag.text.strip() if votes_tag else "0"

                    # --- 5. 时间 (Time) ---
                    # 结构: <div class="m-reviewUnit_userText_footerLane_updated">投稿 : 2025/01/22</div>
                    time_tag = block.select_one(".m-reviewUnit_userText_footerLane_updated")
                    if time_tag:
                        raw_time = time_tag.text.strip()
                        # 清洗数据：去掉 "投稿 : " 前缀
                        review_time = raw_time.replace("投稿 :", "").strip()
                    else:
                        review_time = ""

                    # 只有当评论不为空时才存入
                    if comment:
                        reviews_data.append({
                            "username": username,
                            "rating": rating,
                            "time": review_time,
                            "votes": votes,
                            "comment": comment
                        })

                        if len(reviews_data) >= target_count:
                            break
                
                except Exception as e:
                    # 忽略单条解析错误，继续下一条
                    continue
            
            # 翻页逻辑
            page += 1
            # 随机延时 1.5 - 3.5 秒，防止封禁
            time.sleep(random.uniform(1.5, 3.5))

        except Exception as e:
            print(f"网络请求错误: {e}")
            time.sleep(5) # 出错后多休息一会再试

    return pd.DataFrame(reviews_data)

# --- 主程序入口 ---
if __name__ == "__main__":
    anime_id =11699  # 这里可以替换为你想爬取的动画ID,12093  9062 11699
    target = 100    # 目标数量
    
    df = scrape_anikore_reviews(anime_id, target_count=target)
    
    if not df.empty:
        print(f"\n爬取成功！共获取 {len(df)} 条数据。")
        print("前5条数据预览：")
        print(df.head())
        
        # 保存为 CSV，使用 utf-8-sig 防止日语乱码
        filename = f"anikore_reviews_{anime_id}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"文件已保存为: {filename}")
    else:
        print("未能获取数据。")