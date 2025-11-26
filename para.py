import torch
import numpy as np

def count_parameters(model):
    """
    计算模型总参数量
    """
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """
    计算模型可训练参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """
    打印模型各层参数详情
    """
    total_params = 0
    trainable_params = 0
    
    print("{:<30} {:<20} {:<20}".format("Layer Name", "Parameters", "Trainable"))
    print("-" * 70)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        trainable = param.requires_grad
        total_params += param_count
        if trainable:
            trainable_params += param_count
            
        print("{:<30} {:<20} {:<20}".format(
            name[:30], 
            str(param_count), 
            "Yes" if trainable else "No"
        ))
    
    print("-" * 70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")

def load_and_analyze_model(model_path, model_class=None):
    """
    加载模型并分析参数
    
    Args:
        model_path (str): 模型文件路径
        model_class (class, optional): 模型类定义（如果需要）
    """
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 如果提供了模型类定义
        if model_class is not None:
            model = model_class()
            model.load_state_dict(checkpoint)
        else:
            # 如果保存的是完整模型
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                # 尝试直接加载状态字典（需要提供模型结构）
                raise ValueError("需要提供模型类定义或者保存完整模型")
        
        print(f"成功加载模型: {model_path}")
        print(f"模型类型: {type(model).__name__}")
        print()
        
        # 计算参数量
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"不可训练参数量: {total_params - trainable_params:,}")
        print()
        
        # 显示详细信息
        show_details = input("是否显示各层参数详情? (y/n): ")
        if show_details.lower() == 'y':
            print_model_summary(model)
            
        return model
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 示例1: 加载完整模型
    # model = load_and_analyze_model("model.pth")
    
    # 示例2: 加载状态字典（需要提供模型类）
    # from your_model_file import YourModel
    # model = load_and_analyze_model("model.pth", YourModel)
    
    # 简单使用方式
    import argparse
    
    parser = argparse.ArgumentParser(description='分析PyTorch模型参数量')
    parser.add_argument('model_path', type=str, help='模型文件路径')
    parser.add_argument('--model_class', type=str, help='模型类名称（如果需要）')
    
    args = parser.parse_args()
    
    # 如果提供了模型类名称，需要在代码中定义或导入
    model_class = None
    if args.model_class:
        # 这里需要根据实际情况导入模型类
        # 例如: from models import ResNet
        pass
    
    load_and_analyze_model(args.model_path, model_class)