import numpy as np

def sw_mode(fold):
    classes = np.unique(fold)
    post_pred = []
    
    for data in fold:
        class0_cnt = data.tolist().count(classes[0])
        class1_cnt = data.tolist().count(classes[1])
        
        if class0_cnt>class1_cnt:
            post_pred.append(classes[0])
        else:
            post_pred.append(classes[1])

    return np.array(post_pred)


def sw_lcr(fold):
    classes = np.unique(fold)
    post_pred = []

    for data in fold: 
        mystr = ''.join(str(e) for e in data)
        current_seq_len = 0
        last_char = ""
        max_seq_len = 0
        max_char = mystr[0]

        for c in mystr:
            if c == last_char:
                current_seq_len += 1
                if current_seq_len > max_seq_len:
                    max_seq_len = current_seq_len
                    max_char = c
            else:
                current_seq_len = 1
                last_char = c
        post_pred.append(int(max_char))
        
    return np.array(post_pred)