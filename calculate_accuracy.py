import sed_eval




r=[ {'reference_file':'event_list_devtest_gunshot.txt','estimated_file':'final_report_sorted.txt'}]
for file_pair in r:
    reference_event_list = sed_eval.io.load_event_list(file_pair['reference_file'])
    estimated_event_list = sed_eval.io.load_event_list(file_pair['estimated_file'])
    
event_based_metrics = sed_eval.sound_event.EventBasedMetrics(event_label_list=['gunshot'],
                                                             t_collar=0.500,evaluate_offset=False)

#Calculating metrics file-wise
for i in range(0,len(estimated_event_list)):
    event_based_metrics.evaluate(reference_event_list[i:i+1],estimated_event_list[i:i+1])
    
print(event_based_metrics)
with open('accuracy_report.txt','w') as f:
    f.write(str(event_based_metrics))
