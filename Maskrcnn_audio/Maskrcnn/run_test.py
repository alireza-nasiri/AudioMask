from inference import *
import datetime


def main():
    test_image_path = '../datasets/combine_test_500_glassbreak_preprocessing/'
    
    '''
    The model_pathes should be changed
    '''
    model_pathes = []
    #for i in range(91,100):
    #    model_pathes.append('./logs/bowl201820180721T1946/mask_rcnn_bowl2018_00'+str(i)+'.h5')
    model_pathes = ['./logs/bowl201820180721T1946/mask_rcnn_bowl2018_0035.h5']
    for model_path in model_pathes:
        
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        result_dir = time
        predict(test_image_path, result_dir, model_path, event='glassbreak', generate_prediction_image=False,
                generate_prediction_rle=False, submission_file=time + '/final_result.txt')


if __name__ == "__main__":
    main()
