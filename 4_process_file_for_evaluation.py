


with open('final_report.txt', 'r') as file:
    contents = file.read()
file.close()
lines = contents.split('\n')
l = lines[0:len(lines)]
l.sort()
with open('final_report_sorted.txt','w') as file:
    for s in l:
        if s!='':
            file.write(s+'\n')
