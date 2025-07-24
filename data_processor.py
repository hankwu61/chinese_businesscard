#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文名片數據處理模塊
用於生成和處理中文名片的NER訓練數據
"""

import random
import json
from typing import List, Tuple, Dict

class BusinessCardDataProcessor:
    """中文名片數據處理器"""
    
    def __init__(self):
        # 中文姓名數據
        self.names = [
            "張三", "李四", "王五", "陳六", "林七", "黃八", "吳九", "劉十",
            "趙一", "孫二", "周十一", "吳十二", "鄭十三", "王十四", "馮十五",
            "陳十六", "褚十七", "衛十八", "蔣十九", "沈二十", "韓二十一",
            "楊二十二", "朱二十三", "秦二十四", "尤二十五", "許二十六",
            "何二十七", "呂二十八", "施二十九", "張三十", "孔三十一",
            "曹三十二", "嚴三十三", "華三十四", "金三十五", "魏三十六",
            "陶三十七", "姜三十八", "戚三十九", "謝四十", "鄒四十一",
            "喻四十二", "柏四十三", "水四十四", "竇四十五", "章四十六",
            "雲四十七", "蘇四十八", "潘四十九", "葛五十", "奚五十一"
        ]
        
        # 職稱數據
        self.titles = [
            "總經理", "副總經理", "經理", "副理", "主任", "副主任",
            "專員", "助理", "顧問", "總監", "副總監", "主管",
            "工程師", "高級工程師", "資深工程師", "技術總監",
            "業務經理", "行銷經理", "財務經理", "人資經理",
            "執行長", "營運長", "財務長", "技術長", "行銷長",
            "董事長", "副董事長", "董事", "監事", "秘書長"
        ]
        
        # 公司名稱數據
        self.companies = [
            "依林一集團", "科技股份有限公司", "資訊科技有限公司", "貿易有限公司",
            "建設股份有限公司", "金融控股公司", "保險股份有限公司", "證券股份有限公司",
            "銀行股份有限公司", "電信股份有限公司", "電力股份有限公司", "石油股份有限公司",
            "鋼鐵股份有限公司", "汽車股份有限公司", "電子股份有限公司", "半導體股份有限公司",
            "生物科技股份有限公司", "製藥股份有限公司", "食品股份有限公司", "飲料股份有限公司",
            "服飾股份有限公司", "百貨股份有限公司", "物流股份有限公司", "運輸股份有限公司",
            "航空股份有限公司", "航運股份有限公司", "旅遊股份有限公司", "飯店股份有限公司",
            "房地產股份有限公司", "建築股份有限公司", "工程股份有限公司", "顧問股份有限公司"
        ]
        
        # 電話號碼模板
        self.phone_templates = [
            "02-{}-{}", "03-{}-{}", "04-{}-{}", "05-{}-{}", "06-{}-{}",
            "07-{}-{}", "08-{}-{}", "037-{}-{}", "038-{}-{}", "039-{}-{}"
        ]
        
        # 手機號碼模板
        self.mobile_templates = [
            "09{}", "09{}", "09{}", "09{}", "09{}"
        ]
        
        # 電子郵件域名
        self.email_domains = [
            "gmail.com", "yahoo.com.tw", "hotmail.com", "outlook.com",
            "example.com", "company.com", "business.com", "corp.com",
            "enterprise.com", "group.com", "holdings.com", "ltd.com"
        ]
        
        # 地址模板
        self.address_templates = [
            "台北市信義區信義路五段7號",
            "台北市大安區忠孝東路四段1號",
            "台北市中山區中山北路二段1號",
            "台北市松山區敦化北路1號",
            "台北市內湖區內湖路一段1號",
            "新北市板橋區文化路一段1號",
            "新北市中和區中和路1號",
            "新北市新莊區新莊路1號",
            "桃園市桃園區中正路1號",
            "台中市西區台灣大道一段1號",
            "台中市北區三民路三段1號",
            "台中市南區復興路一段1號",
            "台南市中西區中正路1號",
            "台南市東區東門路一段1號",
            "高雄市前金區中正路1號",
            "高雄市苓雅區四維路1號"
        ]
        
        # 統一編號模板
        self.tax_id_templates = [
            "{}", "{}", "{}", "{}", "{}"
        ]
        
        # 網站域名
        self.website_domains = [
            "www.example.com", "www.company.com", "www.business.com",
            "www.corp.com", "www.enterprise.com", "www.group.com",
            "www.holdings.com", "www.ltd.com", "www.inc.com"
        ]
    
    def generate_phone(self) -> str:
        """生成電話號碼"""
        template = random.choice(self.phone_templates)
        return template.format(
            f"{random.randint(1000, 9999)}",
            f"{random.randint(1000, 9999)}"
        )
    
    def generate_mobile(self) -> str:
        """生成手機號碼"""
        template = random.choice(self.mobile_templates)
        return template.format(f"{random.randint(10000000, 99999999)}")
    
    def generate_email(self, name: str) -> str:
        """生成電子郵件"""
        # 移除姓名中的空格
        clean_name = name.replace(" ", "")
        domain = random.choice(self.email_domains)
        return f"{clean_name.lower()}@{domain}"
    
    def generate_tax_id(self) -> str:
        """生成統一編號"""
        return f"{random.randint(10000000, 99999999)}"
    
    def generate_website(self) -> str:
        """生成網站"""
        return random.choice(self.website_domains)
    
    def generate_business_card_text(self) -> Tuple[str, List[str]]:
        """生成單張名片文本和對應的標籤"""
        # 隨機選擇組件
        name = random.choice(self.names)
        title = random.choice(self.titles)
        company = random.choice(self.companies)
        phone = self.generate_phone()
        mobile = self.generate_mobile()
        email = self.generate_email(name)
        address = random.choice(self.address_templates)
        tax_id = self.generate_tax_id()
        website = self.generate_website()
        
        # 隨機決定是否包含某些組件
        components = []
        labels = []
        
        # 姓名 (總是包含)
        components.append(name)
        labels.extend(["B-NAME"] + ["I-NAME"] * (len(name) - 1))
        
        # 職稱 (80% 機率包含)
        if random.random() < 0.8:
            components.append(title)
            labels.extend(["B-TITLE"] + ["I-TITLE"] * (len(title) - 1))
        
        # 電話 (90% 機率包含)
        if random.random() < 0.9:
            components.append(phone)
            labels.extend(["B-PHONE"] + ["I-PHONE"] * (len(phone) - 1))
        
        # 手機 (70% 機率包含)
        if random.random() < 0.7:
            components.append(mobile)
            labels.extend(["B-MOBILE"] + ["I-MOBILE"] * (len(mobile) - 1))
        
        # 公司 (85% 機率包含)
        if random.random() < 0.85:
            components.append(company)
            labels.extend(["B-COMPANY"] + ["I-COMPANY"] * (len(company) - 1))
        
        # 電子郵件 (75% 機率包含)
        if random.random() < 0.75:
            components.append(email)
            labels.extend(["B-EMAIL"] + ["I-EMAIL"] * (len(email) - 1))
        
        # 地址 (60% 機率包含)
        if random.random() < 0.6:
            components.append(address)
            labels.extend(["B-ADDRESS"] + ["I-ADDRESS"] * (len(address) - 1))
        
        # 統一編號 (40% 機率包含)
        if random.random() < 0.4:
            components.append(tax_id)
            labels.extend(["B-TAX_ID"] + ["I-TAX_ID"] * (len(tax_id) - 1))
        
        # 網站 (50% 機率包含)
        if random.random() < 0.5:
            components.append(website)
            labels.extend(["B-WEBSITE"] + ["I-WEBSITE"] * (len(website) - 1))
        
        # 組合文本
        text = " ".join(components)
        
        # 重新構建標籤列表，考慮空格
        label_chars = []
        char_idx = 0
        
        for i, component in enumerate(components):
            # 添加組件的標籤
            component_labels = labels[char_idx:char_idx + len(component)]
            label_chars.extend(component_labels)
            char_idx += len(component)
            
            # 如果不是最後一個組件，添加空格標籤
            if i < len(components) - 1:
                label_chars.append("O")
        
        # 確保標籤長度與文本長度匹配
        while len(label_chars) < len(text):
            label_chars.append("O")
        
        label_chars = label_chars[:len(text)]
        
        return text, label_chars
    
    def generate_dataset(self, num_samples: int = 1000) -> Tuple[List[str], List[List[str]]]:
        """生成數據集"""
        texts = []
        labels = []
        
        for _ in range(num_samples):
            text, label = self.generate_business_card_text()
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def save_dataset(self, texts: List[str], labels: List[List[str]], filename: str = "business_card_data.json"):
        """保存數據集到文件"""
        data = {
            "texts": texts,
            "labels": labels
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"數據集已保存到 {filename}")
    
    def load_dataset(self, filename: str = "business_card_data.json") -> Tuple[List[str], List[List[str]]]:
        """從文件加載數據集"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data["texts"], data["labels"] 