
disease_code = [
    '00', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
    'a10', 'a11', 'a12', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
    'b8'
]


disease_name = [
    '정상', '딸기잿빛곰파이병', '딸기흰가루병', '오이노균병', '오이흰가루병', '토마토흰가루병', '토마토잿빛곰파이병',
    '고추탄저병', '고추흰가루병', '파프리카흰가루병', '파프리카잘록병', '시설포도탄저병', '시설포도노균병',
    '냉해피해', '열과', '칼슘결핍', '일소피해', '축과병', '다량원소결핍 (N)', '다량원소결핍 (P)', '다량원소결핍 (K)'
]


def match_name(code):
    for index in range(len(disease_code)):
        if code == disease_code[index]:
            return (disease_name[index])
    return None


if __name__ == "__main__":
    data = []
    
    a = 1
    b = a + 2
    data.append({"name" : a, "confidence" :a })
    data.append({"name" : b, "confidence" :123 })
    
    print(data)
    

