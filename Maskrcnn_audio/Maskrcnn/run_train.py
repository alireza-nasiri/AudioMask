from train import train
def main():
    pretrain_model = ''
    log_path = './logs/'
    train_dataset = '../datasets/combine_train_5000_gunshot_Prob1_44100_128_preprocessing'
    val_dataset = '../datasets/combine_val_1000_gunshot_44100_128_preprocessing'
    train(train_dataset, val_dataset, pretrain_model, log_path, num_train=250,
        num_val=95, epochs=50)



if __name__ == "__main__":
    main()
