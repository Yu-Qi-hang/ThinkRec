import yaml
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replaceing")
    parser.add_argument("--ckptdir", type=str, help="path to configuration file.")
    parser.add_argument("--tag", default='', type=str, help="model to be eval.")
    parser.add_argument("--mode", default=0,type=int, help="model to be eval.")

    args = parser.parse_args()

    with open('/home/yuqihang/projects/CoLLM/train_configs/myconfig/reason_mf_book_eval.yaml', 'r') as f:
        data = yaml.safe_load(f)
    if args.mode == 0:
        data['model']['ckpt'] = f"{args.ckptdir}/checkpoint_best{args.tag}.pth"  # 修改字段
    else:
        data['model']['ckpt'] = f"{args.ckptdir}/{args.tag}"  # 修改字段
    data['model']['generate_config']['enable'] = True
    data['model']['prompt_path'] = "prompts/reflection_amazon.txt"
    with open('/home/yuqihang/projects/CoLLM/train_configs/myconfig/reason_mf_book_eval1.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)
    data['model']['generate_config']['enable'] = False
    data['model']['prompt_path'] = "prompts/collm_amazon_.txt"
    with open('/home/yuqihang/projects/CoLLM/train_configs/myconfig/reason_mf_book_eval2.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)