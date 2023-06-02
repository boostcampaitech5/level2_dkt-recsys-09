import os

def select_model_script():
    script_dir = 'test'  # 스크립트가 위치한 폴더명
    
    available_scripts = []  # 사용 가능한 모델 훈련 스크립트 리스트
    for file_name in os.listdir(script_dir):
        if file_name.endswith('.py'):
            available_scripts.append(file_name)
    
    # 사용자로부터 선택 받기
    print("Available model:")
    for i, script in enumerate(available_scripts):
        print(f"{i+1}. {script}")
    
    while True:
        choice = input("Enter the number of the model you want to run: ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(available_scripts):
                selected_script = available_scripts[choice-1]
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return selected_script

if __name__ == '__main__':
    selected_script = select_model_script()
    print(f"Selected script: {selected_script}")
    
    # 선택된 스크립트 실행
    script_path = os.path.join('test', selected_script)
    os.system(f"python {script_path}")