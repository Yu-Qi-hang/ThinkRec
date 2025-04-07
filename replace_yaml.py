import yaml
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replaceing")
    parser.add_argument("--ckptdir", type=str, help="path to configuration file.")

    args = parser.parse_args()

    with open('/home/yuqihang/projects/CoLLM/train_configs/myconfig/reason_mf_book_eval.yaml', 'r') as f:
        data = yaml.safe_load(f)
    data['model']['ckpt'] = f"{args.ckptdir}/checkpoint_best.pth"  # 修改字段
    data['model']['generate_config']['enable'] = True
    data['model']['prompt_path'] = "prompts/reflection_amazon.txt"
    with open('/home/yuqihang/projects/CoLLM/train_configs/myconfig/reason_mf_book_eval1.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)
    data['model']['generate_config']['enable'] = False
    data['model']['prompt_path'] = "prompts/collm_amazon_.txt"
    with open('/home/yuqihang/projects/CoLLM/train_configs/myconfig/reason_mf_book_eval2.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)