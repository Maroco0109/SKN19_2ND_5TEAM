"""

데이터에 대한 처리를 수행하는 모듈

1. 데이터 스플릿 하는 함수
2. 데이터 전처리 클래스
    - 생성 시에 df를 인자로 받고, 모든 전처리 기능은 해당 인자를 기반으로 수행
    - 메소드로 각각의 과정 구현 후 run() 메소드로 일괄 적용
3. 전처리된 데이터 저장 코드

"""

class DataPreprocessing() :
    def __init__(self, df) :
        self.df = df        # 수정할 데이터
        self.raw_data = df  # 수정 전의 원본 데이터를 저장해둠
    
    def drop_cols(self, df=None, cols=None) :
        if df is None :     # 인자로 전달한 데이터가 없으면 생성할 때 넣은 데이터를 활용
            df = self.df    # 모든 전처리 함수를 위와 같은 방식으로 작성
        pass

    """
    모든 전처리 기능 함수로 구현
    """

    def run(self, df=None, encoding='label') :
        self.drop_cols()
        # ...
        return self.df   # 전처리가 완료된 df 반환

def split_data_X_y(df) :        # 데이터를 특성, 라벨(시간, 사건)으로 분리
    pass

def split_data_X_y_e(df) :      # 데이터를 특성, 시간, 사건으로 분리
    pass