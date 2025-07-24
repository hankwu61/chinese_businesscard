#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
預測腳本測試
驗證所有預測腳本都能正常工作
"""

import subprocess
import sys
import os

def test_prediction_script(script_name: str, test_text: str = "張三 總經理 02-1234-5678"):
    """測試預測腳本"""
    print(f"\n{'='*60}")
    print(f"🧪 測試 {script_name}")
    print(f"{'='*60}")
    
    try:
        # 運行預測腳本
        cmd = [sys.executable, script_name, "--text", test_text]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(f"✅ {script_name} 運行成功")
            print("輸出:")
            print(result.stdout)
        else:
            print(f"❌ {script_name} 運行失敗")
            print("錯誤:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {script_name} 測試異常: {e}")
        return False
    
    return True

def main():
    """主測試函數"""
    print("🚀 開始測試所有預測腳本")
    
    test_text = "張三 總經理 02-1234-5678 0912345678 依林一集團 john@example.com"
    
    # 要測試的腳本列表
    scripts = [
        "predict.py",
        "predict_roberta.py", 
        "improved_predict.py"
    ]
    
    success_count = 0
    total_count = len(scripts)
    
    for script in scripts:
        if os.path.exists(script):
            if test_prediction_script(script, test_text):
                success_count += 1
        else:
            print(f"\n❌ 腳本 {script} 不存在")
    
    print(f"\n{'='*60}")
    print(f"📊 測試結果: {success_count}/{total_count} 個腳本測試成功")
    print(f"{'='*60}")
    
    if success_count == total_count:
        print("🎉 所有預測腳本都正常工作！")
        print("\n📝 可用的預測腳本:")
        print("1. python predict.py -- 基礎預測功能")
        print("2. python predict_roberta.py -- RoBERTa預測功能")
        print("3. python improved_predict.py -- 改進版預測功能")
        print("\n💡 使用 --interactive 參數進入互動模式")
    else:
        print("⚠ 部分腳本測試失敗，請檢查錯誤信息")

if __name__ == "__main__":
    main() 