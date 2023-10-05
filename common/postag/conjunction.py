# http://sach.nlv.gov.vn/sach?a=d&d=tdCmRN1940.2.2.12
# 1. thêm vào
ADDITIONAL = {
    "và": "",
    "với": "",
    "cùng": "",
    "cùng với": ""
}

# suy luận
INFERENCE = {
    "hoặc": "",
    "hay": "",
    "hay là": "",
    "hoặc là": ""
}

# 2. kết thúc
END = {
    "thế": "",
    "vậy": "",
    "nên": "",
    "cho nên": "",
    "nên chi": "",
    "vậy nên": "",
    "thành thử": "",
    "bởi thế": "",
    "bởi rứa": "",
    "bởi vậy": "",
    "vì thế": "",
    "vì vậy": "",
    "vậy nên": "",
    "thế nên": "",
    "thành ra": "",
    "bởi vậy nên": "",
    "bởi thế nên": "",
    "vì vậy nên": "",
    "vì thế nên": ""
}

# 3. tăng tiến
INCREASE = {
    "vả lại": "",
    "vả chăng": "",
    "huống": "",
    "huống chi": "",
    "huống hồ": "",
    "phương chi": ""
}

# 4. đối lập
CONTRAST = {
    "nhưng": "",
    "nhưng mà": "",
    "song": "",
    "tuy nhiên": "",
    "thế mà": "",
    "tuy vậy": "",
    "thế nhưng": "",
    "chứ": "",
    "thế nhưng mà": "",
    "vậy mà": ""
}

# 5. chuyển tiếp
CONTINUOUS = {
    "còn như": "",
    "đến như": "",
    "chí như": ""
}

# 6. mục đích
TARGET = {
    "họa chăng": "",
    "ngọ hầu": "",
    "kẻo": "",
    "kẻo lại": "",
    "kẻo mà": "",
    "kẻo nữa": ""
}

# 7. một việc được dự đoán đã xảy ra
RESULT = {
    "hèn nào": "",
    "hèn chi": "",
    "thảo nào": ""
}

# PHỤ THUỘC LIÊN TỪ
# 8. nguyên nhân
CAUSE = {
    "vì": "",
    "bởi": "",
    "bởi vì": "",
    "vì chưng": "",
    "do": "",
    "bởi do": ""
}

# 9. mục đích
TARGET1 = {
    "để": "",
    "để cho": "",
    "để mà": ""
}

# 10. kết thúc, chấm dứt
END1 = {
    "cho đến": "",
    "đến khi": "",
    "đến nỗi": "",
    "đến nước": ""
}

# 11. thời gian
TIME = {
    "khi": "",
    "lúc": "",
    "đang khi": "",
    "đang lúc": "",
    "trong khi": "",
    "trong lúc": "",
    "bao giờ": ""
}

# 12. nhượng bộ
CONCESSION = {
    "dù": "",
    "dẫu": "",
    "dầu": "",
    "tuy": "",
    "tuy rằng": ""
}

# 13.so sánh
COMPARISON = {
    "ví như": "",
    "cầm như": "",
    "cầm bằng": "",
    "cũng như": "",
    "dường như": "",
    "thế nào": "",
    "thế ấy": ""
}

# 14. giả thuyết
ASSUMPTION = {
    "giả": "",
    "phỏng": "",
    "giả sử": "",
    "giá như": "",
    "giá thế": "",
    "phỏng như": "",
    "nếu mà": ""
}

# 15. điều kiện
CONDITION = {
    "hễ": "",
    "nếu": "",
    "ví": "",
    "ví bằng": "",
    "ví chăng": "",
    "ví dù": "",
    "ví thử": "",
    "nhược bằng": ""
}

# 16
OTHER = {
    "rằng": "",
    "dẫu rằng": "",
    "tuy rằng": "",
    "vì rằng": "",
    "mà": "",
    "giá mà": "",
    "vậy mà": "",
    "phỏng mà": "",
    "dẫu mà": "",
    "dù mà": "",
    "khi thì": "",
    "lúc thì": "",
    "rứa thì": "",
    "thế thì": "",
    "vậy thì": "",
    "thì": "",
    "là": "",
    "cùng là": "",
    "hoặc là": "",
    "hay là": "",
    "vậy là": "",
    "thế là": "",
    "rứa là": "",
    "nữa là": "",
    "huống là": "",
    "vì là": "",
    "bởi là": "",
    "bởi vì là": "",
    "vì chung là": "",
    "cũng là": "",
    "như là": "",
    "dẫu là": "",
    "miễn là": "",
    "tuy là": "",
    "lọ là": "",
    "mựa là": "",
    "họa là": "",
    "rất": "",
    "rất là": ""
}

# thuộc về
BELONG = {
    "của": "",
    "thuộc": "",
    "thuộc về": ""
}

# transform each of the above field into list

ADDITIONAL = set(ADDITIONAL.keys())
INFERENCE = set(INFERENCE.keys())
END = set(END.keys())
INCREASE = set(INCREASE.keys())
CONDITION = set(CONDITION.keys())
CONTRAST = set(CONTRAST.keys())
CONTINUOUS = set(CONTINUOUS.keys())
TARGET = set(TARGET.keys())
RESULT = set(RESULT.keys())
CAUSE = set(CAUSE.keys())
TARGET1 = set(TARGET1.keys())
END1 = set(END1.keys())
TIME = set(TIME.keys())
CONCESSION = set(CONCESSION.keys())
COMPARISON = set(COMPARISON.keys())
ASSUMPTION = set(ASSUMPTION.keys())
OTHER = set(OTHER.keys())


CONJUNCTION = ADDITIONAL \
    .union(INFERENCE) \
    .union(END) \
    .union(INCREASE) \
    .union(CONDITION) \
    .union(CONTRAST) \
    .union(CONTINUOUS) \
    .union(TARGET)\
    .union(RESULT)\
    .union(CAUSE)\
    .union(TARGET1)\
    .union(END1)\
    .union(TIME)\
    .union(CONCESSION)\
    .union(COMPARISON)\
    .union(ASSUMPTION)\
    .union(OTHER)
