import torch




def intersection_over_union(box_pred, box_label):
    box_p_x1 = box_pred[...,0:1]
    box_p_y1 = box_pred[...,1:2]
    box_p_x2 = box_pred[...,2:3]
    box_p_y2 = box_pred[...,3:4]


    box_l_x1 = box_label[...,0:1]
    box_l_y1 = box_label[...,1:2]
    box_l_x2 = box_label[...,2:3]
    box_l_y2 = box_label[...,3:4]

    x1 = torch.max(box_p_x1,box_l_x1)
    y1 = torch.max(box_p_y1,box_l_y1)
    x2 = torch.min(box_p_x2,box_l_x2)
    y2 = torch.min(box_p_y2,box_l_y2)

    intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)

    box_p_area = abs((box_p_x2 - box_p_x1) * (box_p_y1 - box_p_y2))
    box_l_area = abs((box_l_x2 - box_l_x1) * (box_l_y1 - box_l_y2))

    return intersection/ (box_p_area + box_l_area - intersection + 1e-6)


