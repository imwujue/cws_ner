import codecs
import re

f1 = "data/testset1/test_cws2.txt"
f2 = "data/process/test_ner2.txt"
f3 = "data/process/test_res_tmp.txt"
f1 = codecs.open(f1, 'r', 'utf-8')
# f3 = codecs.open(f3, 'w', 'utf-8')
# with codecs.open(f2, 'r', 'utf-8') as f2:
#     line2 = f2.readlines()
#     line1 = f1.readlines()
#     # line3 = f3.readlines()
#     for i in range(0,len(line2)):
#         process_line2 = re.sub('\[|\]|sym|bod|tre|tes|dis', '', line2[i])
#         elem1 = re.split('(\s)',line1[i])
#         elem2 = re.split('(\s)',process_line2)
#         elem_real = re.split('(\s)',line2[i])
#         j = 0
#         for i in range(0,len(elem1)):
#             print(i,elem_real)
#             if elem1[i]!=elem2[j]:
#                 f3.write(elem1[i])
#                 print(elem1[i],end="")
#             else:
#                 f3.write(elem_real[i])
#                 print(elem_real[i],end="")
#                 j +=1
# f3.close()
with codecs.open(f3, 'r', 'utf-8') as f3:
    line1 = f1.readlines()
    line3 = f3.readlines()
    for i in range(0,len(line1)):

        line3[i] = re.sub('\[|\]|sym|bod|tre|tes|dis', '', line3[i])
        if line3[i].strip() != line1[i].strip():
            print(line3[i])
            print(i)
            print(line1[i])