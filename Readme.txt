# Readme
模型读取顺序：
首先读取 ./model/pytorch_model.bin
然后读取 finetune的参数  ./output/train_args.bin
然后读取 optimizer与scheduler的参数  ./output/optimizer.pt 与 ./output/scheduler.pt


模型训练注意事项：
如果是从头预训练， 
需要将 config.load_model 设置为 false，
并且更改 model,optimizer,scheduler 的存储路径， 位于 main.py 中 96，99，100行
            model_save_path = os.path.join(output_dir, "BIOS_training_args.bin")
            torch.save(model.state_dict(), model_save_path)
            logging.info("Saving BIOS_model  to %s", output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "BIOS_optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "BIOS_scheduler.pt"))

模型训练的命令：
单卡命令： python3 main-ddp.py --do_train 
ddp命令：  torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py --do_train 

模型预测的命令：
python3 main-dp-inference.py （会自动调用所有的显卡，如果该机器有其他人占用显卡需要更改设置）
模型预测的文件与输出的文件为： do_pred函数的两个传入参数， 位于main-dp-inference.py的146行 


模型预测过后的数据处理：
result_analysis.ipynb


