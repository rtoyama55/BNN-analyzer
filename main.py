import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
from matplotlib import font_manager
font_path1="C:\Windows\Fonts\cambria.ttc"
font_path2="C:\\Windows\\Fonts\\msgothic.ttc"
cambriamath_font=font_manager.FontProperties(fname=font_path1)
msgocic_font=font_manager.FontProperties(fname=font_path2)
from colorama import Fore, Style
import time
np.set_printoptions(suppress=True, precision=3, floatmode='maxprec')

model=0 #0:BNN,1:CA
N=6 #次元数
InitialValue=1 #初期状態
#BNNの初期設定
layer=2 #層の数BNN
select_matrix=0 #使う行列を選択
select_matrix_L2=0
m=0 #シフトパラメータ(0~N-1)
m_L2=0
#CAの初期設定
select_CA=0 #0:ECA,1:MCA
RN=90 #ルールナンバー
RN_mix=[]
Ti=[0,0,0,0,0,0] #しきい値(必ずN個)
Ti_L2=[0,0,0,0,0,0] #小山さんのはTi_L2=-Si
hysteresis=1 #活性化関数をヒステリシスに
#共通の初期設定
sigma=[] #置換識別子(1~N)
horizontal=20 #時空パターンの横軸の最大値(0~2**N-1)
t_start=0
biterror_time_space=[] #[入力する時間(0~horizontal),反転する位置(1~N)]

#出力するグラフ(0:出力しない,1:出力する)
plot_connection=1 #結合パターン
plot_Dmap=1 #Dmap
plot_pattern=1 #時空パターン
plot_energy=0 #エネルギー関数分布

plot_detail=0 #グラフの詳細(タイトル，軸ラベルなど)
plot_skelton=0

#行列を設定
w_1strow=[1,0,-1,0,0,1]  #結合行列の1行目(self_matrix=0)
IN_rows=[125,347,529,447,157,527] #(self_matrix=100) [581,679,469,399,133,287]
w_1strow_2=[1,0,-1,0,0,1] #結合行列の1行目(self_matrix=1)
w_4throw_2=[-1,0,1,1,0,0] #結合行列の4行目(self_matrix=1)
split_w=int(N/2) #結合行列を分割する行
w_allself_2=[[-1, 1],
             [ 1, 1]] #全ての値を自由に設定(self_matrix=2)
w_allself_3=[[ 0, 1, 1],
             [-1, 0,-1],
             [-1, 1, 0]] #全ての値を自由に設定(self_matrix=3)
w_allself_4=[[ 1,-1, 0, 1],
             [ 1, 1,-1, 0],
             [ 0, 1, 1,-1],
             [-1, 0, 1, 1]] #全ての値を自由に設定(self_matrix=4)
w_allself_5=[[-1,-1,-1,-1,-1],
             [-1, 1, 1, 1, 1],
             [ 1,-1, 1, 1, 1],
             [ 1, 1,-1, 1, 1],
             [ 1, 1, 1,-1, 1]] #全ての値を自由に設定(self_matrix=5)
w_allself_6=[[ 1, 0,-1, 0, 0, 1],
             [ 1, 1, 0,-1, 0, 0],
             [ 0, 1, 1, 0, 0,-1], #3行目
             [-1, 0, 1, 1, 0, 0],
             [-1, 0, 0, 1, 1, 0],
             [ 0,-1, 0, 0, 1, 1]] #全ての値を自由に設定(self_matrix=6)
w_allself_7=[[ 1, 0, 0,-1, 0, 0, 1],
             [ 1, 1, 0, 0,-1, 0, 0],
             [ 0, 1, 1, 0, 0,-1, 0],
             [ 0, 0, 1, 1, 0, 0,-1], #4行目
             [-1, 0, 0, 1, 1, 0, 0],
             [ 0,-1, 0, 0, 1, 1, 0],
             [ 0, 0,-1, 0, 0, 1, 1]] #全ての値を自由に設定(self_matrix=7)
w_allself_8=[[ 0, 0,-2,-2, 0, 0, 0, 0],
             [ 0, 0,-2,-2, 0, 0, 4,-4],
             [-2,-2, 0, 4,-2,-2,-2, 2],
             [-2,-2, 4, 0,-2,-2,-2, 2], #4行目
             [ 0, 0,-2,-2, 0, 0, 0, 0],
             [ 0, 0,-2,-2, 0, 0, 0, 0],
             [ 0, 4,-2,-2, 0, 0, 0,-4],
             [ 0,-4, 2, 2, 0, 0,-4, 0]] #全ての値を自由に設定(self_matrix=8)
w_allself_10=[[ 0,-2, 0, 0, 2, 0, 0, 0, 4, 0],
              [-2, 0, 2,-2, 0,-2, 2, 2,-2, 2],
              [ 0, 2, 0,-4, 2, 0, 4, 0, 0, 4],
              [ 0,-2,-4, 0,-2, 0,-4, 0, 0,-4],
              [ 2, 0, 2,-2, 0,-2, 2,-2, 2, 2], #5行目
              [ 0,-2, 0, 0,-2, 0, 0, 0, 0, 0],
              [ 0, 2, 4,-4, 2, 0, 0, 0, 0, 4],
              [ 0, 2, 0, 0,-2, 0, 0, 0, 0, 0],
              [ 4,-2, 0, 0, 2, 0, 0, 0, 0, 0],
              [ 0, 2, 4,-4, 2, 0, 4, 0, 0, 0]] #全ての値を自由に設定(self_matrix=10)
w_1strow_L2=[-1,-1,-1,1,1,1]  #結合行列の1行目(self_matrix=0)
w_1strow_2_L2=[1,0,-1,0,0,1] #結合行列の1行目(self_matrix=1)
w_4throw_2_L2=[-1,0,1,1,0,0] #結合行列の4行目(self_matrix=1)
split_w_L2=int(N/2) #結合行列を分割する行
w_allself_2_L2=[[-1, 1],
                [ 1, 1]] #全ての値を自由に設定(self_matrix=2)
w_allself_3_L2=[[ 0, 1, 1],
                [-1, 0,-1],
                [-1, 1, 0]] #全ての値を自由に設定(self_matrix=3)
w_allself_4_L2=[[ 1,-1, 0, 1],
                [ 1, 1,-1, 0],
                [ 0, 1, 1,-1],
                [-1, 0, 1, 1]] #全ての値を自由に設定(self_matrix=4)
w_allself_5_L2=[[-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1],
                [ 1,-1, 1, 1, 1],
                [ 1, 1,-1, 1, 1],
                [ 1, 1, 1,-1, 1]] #全ての値を自由に設定(self_matrix=5)
w_allself_6_L2=[[ 1,-1,-1, 1,-1, 1],
                [ 1, 1,-1,-1, 1,-1],
                [-1, 1, 1,-1,-1, 1], #3行目
                [-1, 1, 1,-1, 1,-1],
                [-1,-1, 1, 1, 1,-1],
                [-1,-1,-1, 1, 1, 1]] #全ての値を自由に設定(self_matrix=6)
w_allself_7_L2=[[ 1, 0, 0, 0, 0, 0, 0],
                [ 0, 1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 1], #4行目
                [ 0, 0, 1, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 1, 0]] #全ての値を自由に設定(self_matrix=7)
w_allself_8_L2=[[ 1, 0, 0,-1, 0, 0, 0, 1],
                [ 1, 1, 0, 0,-1, 0, 0, 0],
                [ 0, 1, 1, 0, 0,-1, 0, 0],
                [ 0, 0, 1, 1, 0, 0,-1, 0], #4行目
                [ 0, 0, 0, 1, 1, 0, 0,-1],
                [-1, 0, 0, 0, 1, 1, 0, 0],
                [ 0,-1, 0, 0, 0, 1, 1, 0],
                [ 0, 0,-1, 0, 0, 0, 1, 1]] #全ての値を自由に設定(self_matrix=8)
w_allself_10_L2=[[ 1, 0, 0, 0,-1, 0, 0, 0, 0, 1],
                 [ 1, 1, 0, 0, 0,-1, 0, 0, 0, 0],
                 [ 0, 1, 1, 0, 0, 0,-1, 0, 0, 0],
                 [ 0, 0, 1, 1, 0, 0, 0,-1, 0, 0],
                 [ 0, 0, 0, 1, 1, 0, 0, 0,-1, 0], #5行目
                 [-1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [ 0,-1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [ 0, 0,-1, 0, 0, 0, 1, 1, 0, 0],
                 [ 0, 0, 0,-1, 0, 0, 0, 1, 1, 0],
                 [ 0, 0, 0,-1, 0, 0, 0, 0, 1, 1]] #全ての値を自由に設定(self_matrix=10)

row_zero=0 #0にしたい行
row_zero2=0 #0にしたい行
row_zero3=0 #0にしたい行
column_zero=0 #0にしたい列
column_zero2=0 #0にしたい列
column_zero3=0 #0にしたい列
diagonal_zero=0
diagonal_zero2=0
diagonal_zero3=0

row_zero_L2=0 #0にしたい行
row_zero2_L2=0 #0にしたい行
row_zero3_L2=0 #0にしたい行
column_zero_L2=0 #0にしたい列
column_zero2_L2=0 #0にしたい列
column_zero3_L2=0 #0にしたい列
diagonal_zero_L2=0
diagonal_zero2_L2=0
diagonal_zero3_L2=0

#プリントするもの(0:プリントしない,1:プリントする)
print_conection_matrix=1 #結合行列
print_cnt_input=1 #入力数
print_cnt_BPO=1 #設定した結合行列の周期軌道(BPO)の数
print_longest_period=1 #設定した結合行列の最長周期
print_longest_period_EPP=1 #最長周期におけるEPPの数
print_most_EPP=0 #設定した結合行列の最多EPP
print_most_EPP_period=0 #最多EPPの周期
print_fixedpoint=1 #設定した結合行列の不動点の数と値
print_fall_fixedpoint=1 #不動点に落ち込む初期値の数と値
print_step_IV=0 #初期値が収束するまでのステップ数
print_eig_val=1
print_orbit=1 #設定した初期値のBPOの周期と値
print_cnt_EPP=1 #設定した初期値のBPOにおけるEPPの数
print_EPP=1 #設定した初期値のBPOにおけるEPPの値
print_cnt_LSP=1 #設定した初期値のBPOにおけるローカル安定点(LSP)の数
print_LSP=1 #設定した初期値のBPOにおけるローカル安定点の値
print_BitError=0 #設定した初期値のBPPにおける1ビットエラーの値
print_cnt_DSP=1 #設定した初期値のBPOにおけるダイレクト安定点(DSP)の数
print_DSP=0 #設定した初期値のBPOにおけるダイレクト安定点の値
print_cnt_GSP=0 #設定した初期値のBPOにおけるグローバル安定点(GSP)の数(EPPと同じ)
print_GSP=0 #設定した初期値のBPOにおけるグローバル安定点の値
print_orbit_others=1 ##設定した初期値のBPO以外のBPO
print_fixedpoint_others=0
print_runtime=1 #処理にかかった時間


#途中の値
print_xin=0 #入力（二進数) 
print_xout_before=0 #出力（二進数)
print_xout_after=0 #出力(二進数)
print_block=0 #時空パターン(二進数)


#結合パターンで表示するもの
plot_connection_emphasis=0 #(0:全部,1~N:そこだけ結合)

#Dmapに表示するもの(0:非表示,1:表示)
plot_allpoint=1 #全ての点(黒)
plot_allorbit=0 #初期値からの軌道(青)
plot_BPO=1 #BPO(赤)
plot_BPP=1 #BPP(赤)
plot_EPP=1 #EPP(青)
plot_LSP=0 #LSP(青)
plot_DSP=0 #DSP(青白)
plot_GSP=0 #GSP(青)
plot_FP=0 #fixed point(赤)
plot_EPP_FP=0
plot_DEPP_FP=0
plot_BPO_others=0

#時空パターンに表示するもの
plot_idx=1
plot_blackspace=1
plot_redframe=1
plot_xout_before=0
plot_xout_after=0
plot_frame_blue=[]#時刻指定
plot_frame_green=[]
plot_frame_orange=[]
plot_frame_red2=[]#範囲指定
plot_frame_blue2=[]



#変数の宣言
w=np.zeros((N,N)) #結合行列98
w_L2=np.zeros((N,N))
xdec=np.zeros(2**N) #十進数のx座標出力114
xin=np.zeros(((2**N,1,N))) #-1,1の入力120
xcal=np.zeros((1,N)) #行列の計算用132
xcal_L2=np.zeros((1,N))
xout=np.zeros(((2**N,1,N))) #-1,1の出力133
xout_L2=np.zeros(((2**N,1,N)))
xout_before=np.zeros(((2**N,1,N))) #-1,1の出力133
ydec=np.zeros(2**N) #十進数のy座標出力161
cnt_fixedpoint=0 #y=x上の点を数える変数168
fixedpoint=np.zeros(2**N) #不動点の値を格納169
fixedpoint=fixedpoint[fixedpoint!=0]
fixedpoint_others=np.zeros(2**N)
fixedpoint_others=fixedpoint_others[fixedpoint_others!=0]
xline=np.zeros(2**N+1) #初期値からの軌道上の点のx座標172
yline=np.zeros(2**N+1) #初期値からの軌道上の点のy座標173
BPPx=np.zeros(2**N) #BPPのx座標を格納190
BPPy=np.zeros(2**N) #BPPのy座標を格納194
BPPx=BPPx[BPPx!=0]
BPPy=BPPy[BPPy!=0]
cnt_BPO=0 #設定した結合行列のBPOの数をカウント199
step_IV=np.zeros(2**N) #収束するまでのステップ数
BPPfromIV=np.zeros(2**N)
EPPx=np.zeros(2**N) #設定した初期値のBPOにおけるEPPのx座標を格納252
EPPx=EPPx[EPPx!=0]
EPPy=np.zeros(2**N) #設定した初期値のBPOにおけるEPPのy座標を格納253
EPPy=EPPx[EPPx!=0]
cnt_EPP=0 #EPPの数をカウント254
same_BPPx2=0 #既出のBPPとの重複数258
bpp=np.zeros(2**N) #全てのBPPを格納し，重複を避ける259
bpp=bpp[bpp!=0]
length_BPPx2=np.zeros(2**N) #その他のBPOの周期をそれぞれ格納262
cnt_BPO_others=0 #その他のBPOの数だけをカウント262
BPPx_others=np.zeros((int((2**N)/2),2**N)) #その他のBPOの[順番,値]264
EPPx_others=np.zeros(int((2**N)/2)) #その他のBPOにおけるEPPの数をそれぞれ格納272
fall_fixedpoint=np.zeros(2**N) #不動点に落ち込む初期値を格納282
fall_fixedpoint=fall_fixedpoint[fall_fixedpoint!=0]
fall_fixedpoint_initial=np.zeros(2**N)
fall_fixedpoint_initial=fall_fixedpoint_initial[fall_fixedpoint_initial!=0]
fall_fixedpoint_others=np.zeros(2**N)
BPPx_binary=np.zeros(N) #設定した初期値におけるBPPの二進数表示308
BitError=np.zeros(N) #1ビットエラーの点(二進数)315
BitError_dec=np.zeros(2**N) #1ビットエラーの点(十進数)326
BitError_dec=BitError_dec[BitError_dec!=0]
LSP=np.zeros(N) #初期値のL安定点329
LSP=LSP[LSP!=0]
DSP=np.zeros(N) #初期値のD安定点337
DSP=DSP[DSP!=0]
E=np.zeros(2**N) #エネルギー関数
DEPP_fixedpoint=[]


time_start=time.time() #ここから処理時間をカウント
time_start_formatted=time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time_start))
if(print_runtime):
    print(f'実行開始:{time_start_formatted}')
    print()




#計算
if(model==0):
    #行列に値を格納
    if(select_matrix==0):
        cnt_input=np.count_nonzero(w_1strow)
        for i in range(N):
            w[i]=np.roll(w_1strow,i) #一つずつ右にシフトさせ，次の行に格納
    elif(select_matrix==1):
        cnt_input1=np.count_nonzero(w_1strow_2)
        cnt_input4=np.count_nonzero(w_4throw_2)
        for i in range(split_w):
            w[i]=np.roll(w_1strow_2,i)
        for i in range(N-split_w):
            w[split_w+i]=np.roll(w_4throw_2,i)
    elif(select_matrix==2): 
        w=w_allself_2 #自分で決めた行列を格納
    elif(select_matrix==3): 
        w=w_allself_3 #自分で決めた行列を格納
    elif(select_matrix==4): 
        w=w_allself_4 #自分で決めた行列を格納
    elif(select_matrix==5): 
        w=w_allself_5 #自分で決めた行列を格納
    elif(select_matrix==6): 
        w=w_allself_6 #自分で決めた行列を格納
    elif(select_matrix==7): 
        w=w_allself_7 #自分で決めた行列を格納
    elif(select_matrix==8): 
        w=w_allself_8 #自分で決めた行列を格納
    elif(select_matrix==10): 
        w=w_allself_10 #自分で決めた行列を格納
    elif(select_matrix==100): 
        # IN_rows から行列 w を復元する
        # 前提: IN_rows は長さ N の配列で、各要素は 0 <= IN_rows[i] < 3**N の整数
        if not isinstance(IN_rows, (list, tuple, np.ndarray)) or len(IN_rows) != N:
            raise ValueError(f"IN_rows の長さが不正です。N={N} に対して len(IN_rows)={len(IN_rows)}")
    
        w = np.zeros((N, N), dtype=int)
        limit = 3**N
    
        for i, W in enumerate(IN_rows):
            W = int(W)
            if W < 0 or W >= limit:
                raise ValueError(f"IN_rows[{i}]={W} は範囲外です (0～{limit-1})")
    
            # W を 3進展開し、0→-1, 1→0, 2→1 に写像
            # ループは下位桁→上位桁だが、行は左（上位桁）から埋める
            row = np.empty(N, dtype=int)
            temp = W
            for j in range(N):
                r = temp % 3
                if r == 0:
                    row[N-1-j] = -1
                elif r == 1:
                    row[N-1-j] = 0
                else:  # r == 2
                    row[N-1-j] = 1
                temp //= 3
    
            w[i] = row
        
    
    if(layer==3):
        if(select_matrix_L2==0):
            cnt_input_L2=np.count_nonzero(w_1strow_L2)
            for i in range(N):
                w_L2[i]=np.roll(w_1strow_L2,i) #一つずつ右にシフトさせ，次の行に格納
        elif(select_matrix_L2==1):
            cnt_input1_L2=np.count_nonzero(w_1strow_2)
            cnt_input4_L2=np.count_nonzero(w_4throw_2)
            for i in range(split_w):
                w_L2[i]=np.roll(w_1strow_2,i)
            for i in range(N-split_w):
                w_L2[split_w+i]=np.roll(w_4throw_2,i)
        elif(select_matrix_L2==2): 
            w_L2=w_allself_2_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==3): 
            w_L2=w_allself_3_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==4): 
            w_L2=w_allself_4_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==5): 
            w_L2=w_allself_5_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==6): 
            w_L2=w_allself_6_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==7): 
            w_L2=w_allself_7_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==8): 
            w_L2=w_allself_8_L2 #自分で決めた行列を格納
        elif(select_matrix_L2==10): 
            w_L2=w_allself_10_L2 #自分で決めた行列を格納
    
    #0にしたいところを0にする
    if(row_zero>0):
        for i in range(N):
         w[row_zero-1][i]=0  
    if(row_zero2>0):
        for i in range(N):
            w[row_zero2-1][i]=0 
    if(row_zero3>0):
        for i in range(N):
            w[row_zero3-1][i]=0 
    if(column_zero>0):
        for i in range(N):
            w[i][column_zero-1]=0
    if(column_zero2>0):
        for i in range(N):
            w[i][column_zero2-1]=0
    if(column_zero3>0):
        for i in range(N):
            w[i][column_zero3-1]=0
    if(diagonal_zero>0):
        for i in range(N):
            w[i][int((i+diagonal_zero-1)%N)]=0
    if(diagonal_zero2>0):
        for i in range(N):
            w[i][int((i+diagonal_zero2-1)%N)]=0
    if(diagonal_zero3>0):
        for i in range(N):
            w[i][int((i+diagonal_zero3-1)%N)]=0  
    
    if(row_zero_L2>0):
        for i in range(N):
         w_L2[row_zero_L2-1][i]=0  
    if(row_zero2_L2>0):
        for i in range(N):
            w_L2[row_zero2_L2-1][i]=0 
    if(row_zero3_L2>0):
        for i in range(N):
            w_L2[row_zero3_L2-1][i]=0 
    if(column_zero_L2>0):
        for i in range(N):
            w_L2[i][column_zero_L2-1]=0
    if(column_zero2_L2>0):
        for i in range(N):
            w_L2[i][column_zero2_L2-1]=0
    if(column_zero3_L2>0):
        for i in range(N):
            w_L2[i][column_zero3_L2-1]=0
    if(diagonal_zero_L2>0):
        for i in range(N):
            w_L2[i][int((i+diagonal_zero_L2-1)%N)]=0
    if(diagonal_zero2_L2>0):
        for i in range(N):
            w_L2[i][int((i+diagonal_zero2_L2-1)%N)]=0
    if(diagonal_zero3_L2>0):
        for i in range(N):
            w_L2[i][int((i+diagonal_zero3_L2-1)%N)]=0  
    
    
    #シフトパラメータ
    for i in range(N):
        w[i]=np.roll(w[i],-m)
    
    if(layer==3):
        for i in range(N):
            w_L2[i]=np.roll(w_L2[i],-m_L2)
        if(sigma):
            for i in range(N):
                for j in range(N):
                    if(j==sigma[i]-1):
                        w_L2[i][j]=1
                    else:
                        w_L2[i][j]=0
    
    #行列の計算
    if(layer==2):
        eig_val,eig_vec=LA.eig(w) #固有値, 固有ベクトル
        
        for i in range(2**N):
            xdec[i]=i #x座標の十進数
           
            temp_i=i
            #-1と1にして配列に格納(0~63まで)
            for j in range(N):
                if(temp_i%2==0): #余りがないとき
                    xin[i][0][N-j-1]=-1
                    temp_i=temp_i//2
                else: #余りがあるとき
                    xin[i][0][N-j-1]=1
                    temp_i=temp_i//2
            
            if(print_xin):
                print(i,xin[i])
            xin_i = xin[i].reshape((N,1))  # (1,6) → (6,1) に変換
            E[i] = -(1/2) * np.dot(np.dot(xin_i.T, w), xin_i)+np.dot(Ti,xin_i)
            #print(i,E[i])
                
            #行列の掛け算
            for j in range(N):
                for k in range(N):
                    xcal[0][k]=w[j][k]*xin[i][0][k]
                xout[i][0][j]=sum(xcal[0])
                xout_before[i][0][j]=xout[i][0][j]
                
            if(print_xout_before):
                print(i,xout[i])  
                
            #-1と1に変換(シグナム活性化関数)
            if(hysteresis==0):
                for j in range(N):
                    if(xout[i][0][j]>=Ti[j]):
                        xout[i][0][j]=1
                    elif(xout[i][0][j]<Ti[j]):
                        xout[i][0][j]=-1
            elif(hysteresis==1): #ヒステリシスを考慮
                for j in range(N):
                    v = xout[i][0][j]     # 内部電位
                    s = xin[i][0][j]      # 入力（±1 が保証されている）
                
                    t = Ti[j]
                    up   = t
                    down = -t
                
                    if s == 1:
                        # 入力が +1 のとき
                        if v <= down:
                            xout[i][0][j] = -1
                        else:
                            xout[i][0][j] = 1
                    else:  # s == -1
                        # 入力が -1 のとき
                        if v >= up:
                            xout[i][0][j] = 1
                        else:
                            xout[i][0][j] = -1                    
                    
                    #if(xout[i][0][j]>=Ti[j]):
                        #xout[i][0][j]=1
                    #elif(xout[i][0][j]<=-Ti[j]):
                        #xout[i][0][j]=-1
                    #elif(abs(xout[i][0][j])<Ti[j]):
                        #xout[i][0][j]=xin[i][0][j]
                        #if(xin[i][0][j]>Ti[j]):
                            #xout[i][0][j]=1
                        #elif(xin[i][0][j]<Ti[j]):
                            #xout[i][0][j]=-1
            
            if(print_xout_after):
                print(i,xout[i])  
            
            #十進数に変換
            ydec[i]=0#y座標の十進数
            for j in range(N):
                if(xout[i][0][N-j-1]==1):
                    ydec[i]=ydec[i]+xout[i][0][N-j-1]*(2**j)
            
            #y=x上にある点をカウント
            if(xdec[i]==ydec[i]):
                cnt_fixedpoint=cnt_fixedpoint+1
                fixedpoint=np.append(fixedpoint,xdec[i])
    elif(layer==3):
        eig_val,eig_vec=LA.eig(w) #固有値, 固有ベクトル
        eig_val_L2,eig_vec_L2=LA.eig(w_L2)
        
        for i in range(2**N):
            xdec[i]=i #x座標の十進数
           
            temp_i=i
            #-1と1にして配列に格納(0~63まで)
            for j in range(N):
                if(temp_i%2==0): #余りがないとき
                    xin[i][0][N-j-1]=-1
                    temp_i=temp_i//2
                else: #余りがあるとき
                    xin[i][0][N-j-1]=1
                    temp_i=temp_i//2
            
            if(print_xin):
                print(i,xin[i])
                
            #行列の掛け算
            for j in range(N):
                for k in range(N):
                    xcal[0][k]=w[j][k]*xin[i][0][k]
                xout_L2[i][0][j]=sum(xcal[0])
                xout_before[i][0][j]=xout_L2[i][0][j]
            #print(i,xout_L2[i])
            #-1と1に変換(シグナム活性化関数)
            if(hysteresis==0):
                for j in range(N):
                    if(xout_L2[i][0][j]>=Ti[j]):
                        xout_L2[i][0][j]=1
                    elif(xout_L2[i][0][j]<Ti[j]):
                        xout_L2[i][0][j]=-1
            elif(hysteresis==1): #ヒステリシスを考慮    
                for j in range(N):
                    v = xout_L2[i][0][j]     # 内部電位
                    s = xin[i][0][j]      # 入力（±1 が保証されている）
                
                    t = Ti[j]
                    up   = t
                    down = -t
                
                    if s == 1:
                        # 入力が +1 のとき
                        if v <= down:
                            xout_L2[i][0][j] = -1
                        else:
                            xout_L2[i][0][j] = 1
                    else:  # s == -1
                        # 入力が -1 のとき
                        if v >= up:
                            xout_L2[i][0][j] = 1
                        else:
                            xout_L2[i][0][j] = -1
            
            #行列の掛け算
            for j in range(N):
                for k in range(N):
                    xcal_L2[0][k]=w_L2[j][k]*xout_L2[i][0][k]
                xout[i][0][j]=sum(xcal_L2[0])
                xout_before[i][0][j]=xout[i][0][j]
                
            #-1と1に変換(シグナム活性化関数)
            if(hysteresis==0):
                for j in range(N):
                    if(xout[i][0][j]>=Ti_L2[j]):
                        xout[i][0][j]=1
                    elif(xout[i][0][j]<Ti_L2[j]):
                        xout[i][0][j]=-1
            elif(hysteresis==1): #ヒステリシスを考慮
                for j in range(N):
                    v = xout[i][0][j]     # 内部電位
                    s = xout_L2[i][0][j]      # 入力（±1 が保証されている）
                
                    t = Ti_L2[j]
                    up   = t
                    down = -t
                
                    if s == 1:
                        # 入力が +1 のとき
                        if v <= down:
                            xout[i][0][j] = -1
                        else:
                            xout[i][0][j] = 1
                    else:  # s == -1
                        # 入力が -1 のとき
                        if v >= up:
                            xout[i][0][j] = 1
                        else:
                            xout[i][0][j] = -1  
            #print(xout[i])
            #十進数に変換
            ydec[i]=0#y座標の十進数
            for j in range(N):
                if(xout[i][0][N-j-1]==1):
                    ydec[i]=ydec[i]+xout[i][0][N-j-1]*(2**j)
            
            #y=x上にある点をカウント
            if(xdec[i]==ydec[i]):
                cnt_fixedpoint=cnt_fixedpoint+1
                fixedpoint=np.append(fixedpoint,xdec[i])

  
elif(model==1):
    if(select_CA==0):
        rn=np.zeros(8)
        temp_RN=RN
        for i in range(8):
            if(temp_RN%2==0): #余りがないとき
                rn[i]=-1
                temp_RN=temp_RN//2
            else: #余りがあるとき
                rn[i]=1
                temp_RN=temp_RN//2
                
        #ブール関数の定義
        def bool(x1,x2,x3):
            if(x1==1 and x2==1 and x3==1):
                y=rn[7]
            elif(x1==1 and x2==1 and x3==-1):
                y=rn[6]
            elif(x1==1 and x2==-1 and x3==1):
                y=rn[5]
            elif(x1==1 and x2==-1 and x3==-1):
                y=rn[4]
            elif(x1==-1 and x2==1 and x3==1):
                y=rn[3]
            elif(x1==-1 and x2==1 and x3==-1):
                y=rn[2]
            elif(x1==-1 and x2==-1 and x3==1):
                y=rn[1]
            elif(x1==-1 and x2==-1 and x3==-1):
                y=rn[0]
            return y
        
         
        for i in range(2**N):
            xdec[i]=i #x座標の十進数
           
            temp_i=i
            #-1と1にして配列に格納(0~63まで)
            for j in range(N):
                if(temp_i%2==0): #余りがないとき
                    xin[i][0][N-j-1]=-1
                    temp_i=temp_i//2
                else: #余りがあるとき
                    xin[i][0][N-j-1]=1
                    temp_i=temp_i//2
            
            if(print_xin):
                print(i,xin[i])
        
        for i in range(2**N):
            if(sigma):
                for j in range(N):
                    if (j==0):
                        xout_L2[i][0][j]=bool(xin[i][0][N-1],xin[i][0][0],xin[i][0][1])
                    elif (j==N-1):
                        xout_L2[i][0][j]=bool(xin[i][0][N-2],xin[i][0][N-1],xin[i][0][0])
                    else:
                        xout_L2[i][0][j]=bool(xin[i][0][j-1],xin[i][0][j],xin[i][0][j+1])
                for j in range(N):
                    xout[i][0][j]=xout_L2[i][0][int(sigma[j]-1)]
            else:
                for j in range(N):
                    if (j==0):
                        xout[i][0][j]=bool(xin[i][0][N-1],xin[i][0][0],xin[i][0][1])
                    elif (j==N-1):
                        xout[i][0][j]=bool(xin[i][0][N-2],xin[i][0][N-1],xin[i][0][0])
                    else:
                        xout[i][0][j]=bool(xin[i][0][j-1],xin[i][0][j],xin[i][0][j+1])
                        
            #十進数に変換
            ydec[i]=0#y座標の十進数
            for j in range(N):
                if(xout[i][0][N-j-1]==1):
                    ydec[i]=ydec[i]+xout[i][0][N-j-1]*(2**j)
            
            #y=x上にある点をカウント
            if(xdec[i]==ydec[i]):
                cnt_fixedpoint=cnt_fixedpoint+1
                fixedpoint=np.append(fixedpoint,xdec[i])
    
    elif(select_CA==1):
        rn1=np.zeros(8)
        rn2=np.zeros(8)
        temp_RN1=RN_mix[0]
        temp_RN2=RN_mix[1]
        for i in range(8):
            if(temp_RN1%2==0): #余りがないとき
                rn1[i]=-1
                temp_RN1=temp_RN1//2
            else: #余りがあるとき
                rn1[i]=1
                temp_RN1=temp_RN1//2
        for i in range(8):
            if(temp_RN2%2==0): #余りがないとき
                rn2[i]=-1
                temp_RN2=temp_RN2//2
            else: #余りがあるとき
                rn2[i]=1
                temp_RN2=temp_RN2//2
            
        def bool1(x1,x2,x3):
            if(x1==1 and x2==1 and x3==1):
                y=rn1[7]
            elif(x1==1 and x2==1 and x3==-1):
                y=rn1[6]
            elif(x1==1 and x2==-1 and x3==1):
                y=rn1[5]
            elif(x1==1 and x2==-1 and x3==-1):
                y=rn1[4]
            elif(x1==-1 and x2==1 and x3==1):
                y=rn1[3]
            elif(x1==-1 and x2==1 and x3==-1):
                y=rn1[2]
            elif(x1==-1 and x2==-1 and x3==1):
                y=rn1[1]
            elif(x1==-1 and x2==-1 and x3==-1):
                y=rn1[0]
            return y
    
        def bool2(x1,x2,x3):
            if(x1==1 and x2==1 and x3==1):
                y=rn2[7]
            elif(x1==1 and x2==1 and x3==-1):
                y=rn2[6]
            elif(x1==1 and x2==-1 and x3==1):
                y=rn2[5]
            elif(x1==1 and x2==-1 and x3==-1):
                y=rn2[4]
            elif(x1==-1 and x2==1 and x3==1):
                y=rn2[3]
            elif(x1==-1 and x2==1 and x3==-1):
                y=rn2[2]
            elif(x1==-1 and x2==-1 and x3==1):
                y=rn2[1]
            elif(x1==-1 and x2==-1 and x3==-1):
                y=rn2[0]
            return y
        
        for i in range(2**N):
            xdec[i]=i #x座標の十進数      
            temp_i=i
            #-1と1にして配列に格納(0~63まで)
            for j in range(N):
                if(temp_i%2==0): #余りがないとき
                    xin[i][0][N-j-1]=-1
                    temp_i=temp_i//2
                else: #余りがあるとき
                    xin[i][0][N-j-1]=1
                    temp_i=temp_i//2
            
            if(print_xin):
                print(i,xin[i])
        
        for i in range(2**N):
            if(sigma):
                for j in range(N):
                    if(j==0):
                        xout_L2[i][0][j]=bool1(xin[i][0][N-1],xin[i][0][0],xin[i][0][1])
                    elif(j<(N/2)):
                        xout_L2[i][0][j]=bool1(xin[i][0][j-1],xin[i][0][j],xin[i][0][j+1])
                    elif(j<N-1):
                        xout_L2[i][0][j]=bool2(xin[i][0][j-1],xin[i][0][j],xin[i][0][j+1])
                    elif(j==N-1):
                        xout_L2[i][0][j]=bool2(xin[i][0][N-2],xin[i][0][N-1],xin[i][0][0])
                for j in range(N):
                    xout[i][0][j]=xout_L2[i][0][int(sigma[j]-1)]
            else:
                for j in range(N):
                    if(j==0):
                        xout[i][0][j]=bool1(xin[i][0][N-1],xin[i][0][0],xin[i][0][1])
                    elif(j<(N/2)):
                        xout[i][0][j]=bool1(xin[i][0][j-1],xin[i][0][j],xin[i][0][j+1])
                    elif(j<N-1):
                        xout[i][0][j]=bool2(xin[i][0][j-1],xin[i][0][j],xin[i][0][j+1])
                    elif(j==N-1):
                        xout[i][0][j]=bool2(xin[i][0][N-2],xin[i][0][N-1],xin[i][0][0])
                        
            #十進数に変換
            ydec[i]=0#y座標の十進数
            for j in range(N):
                if(xout[i][0][N-j-1]==1):
                    ydec[i]=ydec[i]+xout[i][0][N-j-1]*(2**j)
            
            #y=x上にある点をカウント
            if(xdec[i]==ydec[i]):
                cnt_fixedpoint=cnt_fixedpoint+1
                fixedpoint=np.append(fixedpoint,xdec[i])
    
#初期値を含む周期軌道
xline[0]=xdec[InitialValue]
yline[0]=ydec[InitialValue]

if(biterror_time_space):
    for i in range(int(biterror_time_space[0])):
        for j in range(2**N):
            if(yline[i]==xdec[j]):
                xline[i+1]=xdec[j]
                yline[i+1]=ydec[j]
    
    temp_xin=xin[int(xline[int(biterror_time_space[0])])].copy()
    if(temp_xin[0][int(biterror_time_space[1]-1)]>0):
        temp_xin[0][int(biterror_time_space[1]-1)]=-1
    else:
        temp_xin[0][int(biterror_time_space[1]-1)]=1
    xline[int(biterror_time_space[0])]=0
    for i in range(N):
        if(temp_xin[0][N-i-1]==1):
            xline[int(biterror_time_space[0])]=xline[int(biterror_time_space[0])]+temp_xin[0][N-i-1]*(2**i)
    yline[int(biterror_time_space[0])]=ydec[int(xline[int(biterror_time_space[0])])].copy()
    
    for i in range(int(2**N-biterror_time_space[0])):
        temp_i=int(i+biterror_time_space[0])
        for j in range(2**N):
            if(yline[temp_i]==xdec[j]):
                xline[temp_i+1]=xdec[j]
                yline[temp_i+1]=ydec[j]

else:
    for i in range(2**N):
        for j in range(2**N):
            if(yline[i]==xdec[j]):
                xline[i+1]=xdec[j]
                yline[i+1]=ydec[j]

#BPPを格納
if(biterror_time_space):
    for i in range(int(biterror_time_space[0])+1,2**N-int(biterror_time_space[0])):
        for j in range(i+1,2**N+1-int(biterror_time_space[0])):
            if(xline[i]==xline[j]):
                BPPx=xline[i:j]
                BPPy=yline[i:j]
                
                break
        if(BPPx.size>0):
            break
else:
    for i in range(2**N):
        for j in range(i+1,2**N+1):
            if(xline[i]==xline[j]):
                BPPx=xline[i:j]
                BPPy=yline[i:j]
                break
        if(BPPx.size>0):
            break

notBPP=2**N-len(BPPx)
P=len(BPPx)
longest_period=P #最長周期の初期値

if(P>1):
    cnt_BPO=cnt_BPO+1
    
#配列の要素の重複数をカウントする関数を定義
def count(arr1,arr2):
    # 配列をフラットにしてから、重複を削除して同じ要素を取得
    same_elements=np.intersect1d(arr1.flatten(),arr2.flatten())
    # 同じ要素の数を返す
    return len(same_elements)

fixedpoint_others = fixedpoint[fixedpoint != BPPx[0]]

#全ての初期値における周期点を調べる
for IV in range(2**N):  
    xline2=np.zeros(2**N+1) #その他の初期値からの軌道上の点のx座標
    yline2=np.zeros(2**N+1) #その他の初期値からの軌道上の点のy座標
    BPPx2=np.zeros(2**N)#全ての初期値におけるBPPのx座標
    BPPy2=np.zeros(2**N)#全ての初期値におけるBPPのy座標
    BPPx2=BPPx2[BPPx2!=0]
    BPPy2=BPPy2[BPPy2!=0]
    same=np.zeros(2**N) #設定した初期値におけるBPPとその他の初期値におけるBPPの重複数
 
    xline2[0]=xdec[IV]
    yline2[0]=ydec[IV]
    
    for i in range(2**N):
        for j in range(2**N):
            if(yline2[i]==xdec[j]):
                xline2[i+1]=xdec[j]
                yline2[i+1]=ydec[j]

    #BPPを格納
    for i in range(2**N):
        for j in range(i+1,2**N+1):
            step_IV[IV]=i
            if(xline2[i]==xline2[j]):
                BPPx2=xline2[i:j]
                BPPy2=yline2[i:j]
                break
        if(BPPx2.size>0):
            break
    BPPfromIV[IV]=BPPx2[0]
    
    same[IV]=count(BPPx,BPPx2)
    
    #BPOに落ち込む初期値EPP
    if(same[IV]>1):
        if(xdec[IV] not in BPPx):
            EPPx=np.append(EPPx,xdec[IV])
            EPPy=np.append(EPPy,ydec[IV])
            cnt_EPP=cnt_EPP+1
            
    elif(same[IV]<=1):
        if(len(BPPx2)>1):
            same_BPPx2=count(bpp,BPPx2)
            if(same_BPPx2==0):
                cnt_BPO=cnt_BPO+1
                length_BPPx2[cnt_BPO_others]=len(BPPx2)
                for i in range(len(BPPx2)):
                    BPPx_others[cnt_BPO_others][i]=BPPx2[i]
                cnt_BPO_others=cnt_BPO_others+1
                bpp=np.append(bpp,BPPx2)
                if(len(BPPx2)>longest_period):#最長周期を更新
                    longest_period=len(BPPx2)
                if(IV not in BPPx2):
                    for i in range(cnt_BPO_others):
                        if(count(BPPx2,BPPx_others[i])!=0):
                            EPPx_others[i]= EPPx_others[i]+1
                            break
            else:
                if(IV not in BPPx2):
                    for i in range(cnt_BPO_others):
                        if(count(BPPx2,BPPx_others[i])!=0):
                            EPPx_others[i]= EPPx_others[i]+1
                            break
        else:
            if(IV not in fixedpoint):
                fall_fixedpoint=np.append(fall_fixedpoint,IV)
                for i in range(len(fixedpoint_others)):
                    if(fixedpoint_others[i]==BPPx2):
                        fall_fixedpoint_others[i]=fall_fixedpoint_others[i]+1
                if np.all(BPPx==BPPx2):
                    fall_fixedpoint_initial=np.append(fall_fixedpoint_initial,IV)
                    EPPx=np.append(EPPx,xdec[IV])
                    EPPy=np.append(EPPy,ydec[IV])
        
        if IV not in fixedpoint:
            next_state = int(ydec[IV]) 
            if next_state in fixedpoint:
                DEPP_fixedpoint=np.append(DEPP_fixedpoint,IV)


most_EPP=len(EPPx) #最多のEPPをカウント
most_EPP_period=P #最多のEPPのときの周期を格納

#最長周期のEPPをカウント
cnt_longest_period_EPP=0 #最長周期のEPPの数288
if(P==longest_period):
    cnt_longest_period_EPP=most_EPP
for i in range(cnt_BPO_others):
    if(length_BPPx2[i]==longest_period):
        if(EPPx_others[i]>cnt_longest_period_EPP):
            cnt_longest_period_EPP=EPPx_others[i]
    if(most_EPP<=EPPx_others[i]):
        if(most_EPP_period<=length_BPPx2[i]):
            most_EPP=EPPx_others[i]
            most_EPP_period=length_BPPx2[i]

for i in range(P):
    temp_BPPx=BPPx[i]
    for j in range(N):
        if(temp_BPPx%2==0):
            BPPx_binary[N-j-1]=-1
            temp_BPPx=temp_BPPx//2
        else:
            BPPx_binary[N-j-1]=1
            temp_BPPx=temp_BPPx//2
    
    for j in range(N):
        BitError=BPPx_binary.copy()
        if(BitError[N-j-1]==-1):
            BitError[N-j-1]=1
        else:
            BitError[N-j-1]=-1
        
        temp_BitError_dec=0
        for k in range(N):
            if(BitError[N-k-1]==1):
                temp_BitError_dec=int(temp_BitError_dec+BitError[N-k-1]*(2**k))
        if(temp_BitError_dec not in BPPx):
            BitError_dec=np.append(BitError_dec,temp_BitError_dec)
        if(temp_BitError_dec in EPPx):
            if(temp_BitError_dec not in BPPx):
                if(temp_BitError_dec not in LSP):
                    LSP=np.append(LSP,temp_BitError_dec)             

LSP=np.unique(LSP)
BitError_dec=np.unique(BitError_dec)

for i in range(len(EPPy)):
    if(EPPy[i] in BPPy):
        DSP=np.append(DSP,EPPx[i])

GSP=EPPx #GSPはEPPと同じ





#グラフ
#結合パターンを描画
if(plot_connection):
    if(model==0):
        if(layer==2):
            plt.figure(dpi=300,figsize=(6.85,4))
            plt.xlim(-0.5,N-0.5)
            plt.ylim(-2,7)
            plt.axis('off')
            if(plot_skelton):
                plt.gcf().patch.set_alpha(0)
                plt.gca().patch.set_alpha(0)
            
            for i in range(N):
                if(plot_detail):
                    plt.text(i-0.1,6.3,'$x_{%i}(t+1)$'%(i+1),fontsize=12)
                plt.annotate('', xy=(i,6.1),xytext=(i,5.3),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
                if(plot_detail):
                    #plt.text(i-0.08,4.85,f'$T_{i+1}$')
                    plt.text(i-0.05,4.85,'0')
                
                for j in range(N):
                    if(w[j][i]>=1):
                        plt.plot((i,j),(0.2,4.6),color='red',alpha=0.2,zorder=-1)
                        #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1)
                    elif(w[j][i]<=-1):
                        plt.plot((i,j),(0.2,4.6),color='royalblue',alpha=0.2,zorder=-1)
                        #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1) 
                
                if(plot_connection_emphasis==0):
                    plt.scatter(i,5,color ='green',s=350)
                    plt.scatter(i,5,color=(0.851, 0.949, 0.816),s=250)
                    for j in range(N):
                        if(w[j][i]>=1):
                            plt.plot((i,j),(0.2,4.6),color='red',linewidth=2,zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1)
                        elif(w[j][i]<=-1):
                            plt.plot((i,j),(0.2,4.6),color='royalblue',linewidth=2,zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1) 
                            
                elif(plot_connection_emphasis<=N):
                    plt.scatter(i,5,color ='green',s=350)
                    plt.scatter(i,5,color ='white',s=250)
                    emp=plot_connection_emphasis-1
                    #for j in range(emp):
                        #if(w[j][i]==1):
                            #plt.plot((i,j),(0.2,4.6),color='red',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                        #elif(w[j][i]==-1):
                            #plt.plot((i,j),(0.2,4.6),color='royalblue',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                    
                    if(w[emp][i]>=1):
                        plt.plot((i,emp),(0.2,4.6),color='red',zorder=-1,linewidth=2.5)
                        #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                    elif(w[emp][i]<=-1):
                        plt.plot((i,emp),(0.2,4.6),color='royalblue',zorder=-1,linewidth=2.5)
                        #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1) 
                
                elif(plot_connection_emphasis>N):
                    plt.scatter(i,5,color ='green',s=350,alpha=0.2)
                    plt.scatter(i,5,color ='white',s=250)
                        
                plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-',connectionstyle="arc3",color='black'))
                #plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
                #plt.scatter(i,6.23,color ='black',s=100)
                plt.scatter(i,0,color ='black',s=100)
                if(plot_detail):
                    plt.text(i-0.1,-1.25,f'$x_{i+1}(t)$',fontsize=12)
            plt.show()
        elif(layer==3):
            plt.figure(dpi=300,figsize=(6.85,6))
            plt.xlim(-0.5,N-0.5)
            plt.ylim(-1.5,13)
            plt.axis('off')
            
            for i in range(N):
                plt.annotate('', xy=(i,6.1+6.2),xytext=(i,5.3+6.2),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
                
                for j in range(N):
                    if(w_L2[j][i]>=1):
                        plt.plot((i,j),(0.2+6.2,4.6+6.2),color='red',alpha=0.2,zorder=-1)
                        #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1)
                    elif(w_L2[j][i]<=-1):
                        plt.plot((i,j),(0.2+6.2,4.6+6.2),color='royalblue',alpha=0.2,zorder=-1)
                        #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1) 
                
                if(plot_connection_emphasis==0):
                    plt.scatter(i,5+6.2,color ='green',s=350)
                    plt.scatter(i,5+6.2,color ='white',s=250)
                    for j in range(N):
                        if(w_L2[j][i]>=1):
                            plt.plot((i,j),(0.2+6.2,4.6+6.2),color='red',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1)
                        elif(w_L2[j][i]<=-1):
                            plt.plot((i,j),(0.2+6.2,4.6+6.2),color='royalblue',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1) 
                            
                elif(plot_connection_emphasis<=N):
                    plt.scatter(i,5,color ='green',s=350)
                    plt.scatter(i,5,color ='white',s=250)
                    emp=plot_connection_emphasis-1
                    #for j in range(emp):
                        #if(w_L2[j][i]==1):
                            #plt.plot((i,j),(0.2,4.6),color='red',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                        #elif(w_L2[j][i]==-1):
                            #plt.plot((i,j),(0.2,4.6),color='royalblue',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                    
                    if(w_L2[emp][i]>=1):
                        plt.plot((i,emp),(0.2+6.2,4.6+6.2),color='red',zorder=-1,linewidth=2.5)
                        #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                    elif(w_L2[emp][i]<=-1):
                        plt.plot((i,emp),(0.2+6.2,4.6+6.2),color='royalblue',zorder=-1,linewidth=2.5)
                        #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1) 
                
                elif(plot_connection_emphasis>N):
                    plt.scatter(i,5+6.2,color ='green',s=350,alpha=0.2)
                    plt.scatter(i,5+6.2,color ='white',s=250)
                        
                #plt.annotate('', xy=(i,-0.1+6),xytext=(i,-0.8+6),arrowprops=dict(arrowstyle='-',connectionstyle="arc3",color='black'))
                #plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
                #plt.scatter(i,6.23,color ='black',s=100)
                plt.scatter(i,0+6.2,color ='black',s=100)
            
            
            for i in range(N):
                plt.annotate('', xy=(i,6.1),xytext=(i,5.3),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
                
                for j in range(N):
                    if(w[j][i]==1):
                        plt.plot((i,j),(0.2,4.6),color='red',alpha=0.2,zorder=-1)
                        #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1)
                    elif(w[j][i]==-1):
                        plt.plot((i,j),(0.2,4.6),color='royalblue',alpha=0.2,zorder=-1)
                        #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1) 
                
                if(plot_connection_emphasis==0):
                    plt.scatter(i,5,color ='green',s=350)
                    plt.scatter(i,5,color ='white',s=250)
                    for j in range(N):
                        if(w[j][i]==1):
                            plt.plot((i,j),(0.2,4.6),color='red',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1)
                        elif(w[j][i]==-1):
                            plt.plot((i,j),(0.2,4.6),color='royalblue',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',alpha=0.2,zorder=-1) 
                            
                elif(plot_connection_emphasis<=N):
                    plt.scatter(i,5,color ='green',s=350)
                    plt.scatter(i,5,color ='white',s=250)
                    emp=plot_connection_emphasis-1
                    #for j in range(emp):
                        #if(w[j][i]==1):
                            #plt.plot((i,j),(0.2,4.6),color='red',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                        #elif(w[j][i]==-1):
                            #plt.plot((i,j),(0.2,4.6),color='royalblue',zorder=-1)
                            #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                    
                    if(w[emp][i]==1):
                        plt.plot((i,emp),(0.2,4.6),color='red',zorder=-1,linewidth=2.5)
                        #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1)
                    elif(w[emp][i]==-1):
                        plt.plot((i,emp),(0.2,4.6),color='royalblue',zorder=-1,linewidth=2.5)
                        #plt.plot((i,j),(0.2,4.6),color='black',zorder=-1) 
                
                elif(plot_connection_emphasis>N):
                    plt.scatter(i,5,color ='green',s=350,alpha=0.2)
                    plt.scatter(i,5,color ='white',s=250)
                        
                plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-',connectionstyle="arc3",color='black'))
                #plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
                #plt.scatter(i,6.23,color ='black',s=100)
                plt.scatter(i,0,color ='black',s=100)
            plt.show()
    elif(model==1):
        plt.figure(dpi=300,figsize=(6.85,6))
        plt.xlim(-0.5,N-0.5)
        plt.ylim(-1.5,13)
        plt.axis('off')
        
        for i in range(N):
            plt.annotate('', xy=(i,6.1+6.2),xytext=(i,5.3+6.2),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
            
            if(sigma):
                for j in range(N):
                    plt.plot((sigma[i]-1,i),(0.2+6.2,4.6+6.2),color='black',zorder=-1)
            else:
                for j in range(N):
                    plt.plot((i,i),(0.2+6.2,4.6+6.2),color='black',zorder=-1)
                
            plt.scatter(i,5+6.2,color ='green',s=350)
            plt.scatter(i,5+6.2,color ='white',s=250)
            plt.scatter(i,0+6.2,color ='black',s=100)
        
        
        for i in range(N):
            plt.annotate('', xy=(i,6.1),xytext=(i,5.3),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
            
            plt.scatter(i,5,color ='green',s=350)
            plt.scatter(i,5,color ='white',s=250)
            plt.plot((i,i),(0.2,4.6),color='black',zorder=-1)
            plt.plot((i,(i+1)%N),(0.2,4.6),color='black',zorder=-1)
            plt.plot((i,(i-1)%N),(0.2,4.6),color='black',zorder=-1)
                    
            plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-',connectionstyle="arc3",color='black'))
            #plt.annotate('', xy=(i,-0.1),xytext=(i,-0.8),arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3",color='black'),zorder=-1)
            #plt.scatter(i,6.23,color ='black',s=100)
            plt.scatter(i,0,color ='black',s=100)
        plt.show()



#Dmapを描画
if(plot_Dmap):
    #グラフの体裁
    plt.figure(dpi=300, figsize=(4, 4))
    plt.xlim(-0.05-0.08,(2**N-1)/(2**N)+0.05+0.003)#x軸メモリの(最大,最小)
    plt.ylim(-0.05-0.08,(2**N-1)/(2**N)+0.05+0.003)#y軸メモリの(最大,最小)
    #plt.xticks([0,(2**N)/2,2**N-1],[0,int((2**N)/2),2**N-1])#x軸に表示する場所と値
    #plt.yticks([0,(2**N)/2,2**N-1],[0,int((2**N)/2),2**N-1])#y軸に表示する場所と値
    #plt.xticks([0,((2**N)/2-1)/(2**N),(2**N-1)/(2**N)],['$C_1$',f'$C_{{{int((2**N)/2)}}}$',f'$C_{{{2**N}}}$'],fontsize=10,fontproperties=cambriamath_font)#x軸に表示する場所と値
    #plt.yticks([0,((2**N)/2-1)/(2**N),(2**N-1)/(2**N)],['$C_1$',f'$C_{{{int((2**N)/2)}}}$',f'$C_{{{2**N}}}$'],fontsize=10,fontproperties=cambriamath_font)#y軸に表示する場所と値
    #plt.gca().xaxis.set_minor_locator(ticker.FixedLocator([0,(2**N/8-1)/(2**N),(2**N/4-1)/(2**N),(2**N/8*3-1)/(2**N),(2**N/2-1)/(2**N),(2**N/8*5-1)/(2**N),(2**N/4*3-1)/(2**N),(2**N/8*7-1)/(2**N),(2**N-1)/(2**N)]))
    #}plt.gca().yaxis.set_minor_locator(ticker.FixedLocator([0,(2**N/8-1)/(2**N),(2**N/4-1)/(2**N),(2**N/8*3-1)/(2**N),(2**N/2-1)/(2**N),(2**N/8*5-1)/(2**N),(2**N/4*3-1)/(2**N),(2**N/8*7-1)/(2**N),(2**N-1)/(2**N)]))
    plt.axis('off')
    # ★ Dmap のプロット範囲を白で塗りつぶす（最背面）
    plt.gca().add_patch(
        plt.Rectangle(
            (-0.05, -0.05),                    # 左下 (x,y)
            (2**N - 1)/(2**N) + 0.05 + 0.003 + 0.05,  # 横幅
            (2**N - 1)/(2**N) + 0.05 + 0.003 + 0.05,  # 縦幅
            facecolor='white',
            edgecolor='none',
            zorder=-10
        )
    )
    
    if(plot_skelton):
        plt.gcf().patch.set_alpha(0)
    
    plt.plot((0-0.05,0-0.05),(0-0.05,(2**N-1)/(2**N)+0.05),color="black",linewidth=1)
    plt.plot((0-0.05,(2**N-1)/(2**N)+0.05),((2**N-1)/(2**N)+0.05,(2**N-1)/(2**N)+0.05),color="black",linewidth=1)
    plt.plot(((2**N-1)/(2**N)+0.05,(2**N-1)/(2**N)+0.05),((2**N-1)/(2**N)+0.05,0-0.05),color="black",linewidth=1)
    plt.plot(((2**N-1)/(2**N)+0.05,0-0.05),(0-0.05,0-0.05),color="black",linewidth=1)
    plt.plot((0,0),(-0.05,-0.07),color="black",linewidth=1)#x軸の目盛り
    #plt.text(0,-0.12,"$C_1$",ha="center",va="center",size=10,fontproperties=cambriamath_font)
    plt.plot((((2**N)/2-1)/(2**N),((2**N)/2-1)/(2**N)),(-0.05,-0.07),color="black",linewidth=1)
    #plt.text(((2**N)/2-1)/(2**N),-0.12,f'$C_{{{int((2**N)/2)}}}$',ha="center",va="center",size=10,fontproperties=cambriamath_font)
    plt.plot(((2**N-1)/(2**N),(2**N-1)/(2**N)),(-0.05,-0.07),color="black",linewidth=1)
    #plt.text((2**N-1)/(2**N),-0.12,f'$C_{{{2**N}}}$',ha="center",va="center",size=10,fontproperties=cambriamath_font)
    plt.plot(((2**N/8-1)/(2**N),(2**N/8-1)/(2**N)),(-0.05,-0.06),color="black",linewidth=1)
    plt.plot(((2**N/4-1)/(2**N),(2**N/4-1)/(2**N)),(-0.05,-0.06),color="black",linewidth=1)
    plt.plot(((2**N/8*3-1)/(2**N),(2**N/8*3-1)/(2**N)),(-0.05,-0.06),color="black",linewidth=1)
    plt.plot(((2**N/8*5-1)/(2**N),(2**N/8*5-1)/(2**N)),(-0.05,-0.06),color="black",linewidth=1)
    plt.plot(((2**N/4*3-1)/(2**N),(2**N/4*3-1)/(2**N)),(-0.05,-0.06),color="black",linewidth=1)
    plt.plot(((2**N/8*7-1)/(2**N),(2**N/8*7-1)/(2**N)),(-0.05,-0.06),color="black",linewidth=1)
    
    plt.plot((-0.05,-0.07),(0,0),color="black",linewidth=1)#y軸の目盛り
    #plt.text(-0.12,0,"$C_1$",ha="center",va="center",size=10,fontproperties=cambriamath_font)
    plt.plot((-0.05,-0.07),(((2**N)/2-1)/(2**N),((2**N)/2-1)/(2**N)),color="black",linewidth=1)
    #plt.text(-0.12,((2**N)/2-1)/(2**N),f'$C_{{{int((2**N)/2)}}}$',ha="center",va="center",size=10,fontproperties=cambriamath_font)
    plt.plot((-0.05,-0.07),((2**N-1)/(2**N),(2**N-1)/(2**N)),color="black",linewidth=1)
    #plt.text(-0.12,(2**N-1)/(2**N),f'$C_{{{2**N}}}$',ha="center",va="center",size=10,fontproperties=cambriamath_font)
    plt.plot((-0.05,-0.06),((2**N/8-1)/(2**N),(2**N/8-1)/(2**N)),color="black",linewidth=1)
    plt.plot((-0.05,-0.06),((2**N/4-1)/(2**N),(2**N/4-1)/(2**N)),color="black",linewidth=1)
    plt.plot((-0.05,-0.06),((2**N/8*3-1)/(2**N),(2**N/8*3-1)/(2**N)),color="black",linewidth=1)
    plt.plot((-0.05,-0.06),((2**N/8*5-1)/(2**N),(2**N/8*5-1)/(2**N)),color="black",linewidth=1)
    plt.plot((-0.05,-0.06),((2**N/4*3-1)/(2**N),(2**N/4*3-1)/(2**N)),color="black",linewidth=1)
    plt.plot((-0.05,-0.06),((2**N/8*7-1)/(2**N),(2**N/8*7-1)/(2**N)),color="black",linewidth=1)
        
    if(plot_detail):
        plt.xlabel('$x(t)$',x=0.75/(2**N))#x軸のラベル
        plt.ylabel('$x(t+1)$',y=0.75/(2**N))#y軸のラベル
    x=np.linspace(0,(2**N-1)/(2**N),10)
    y=x
    plt.plot(x,y,color='black',linewidth=1,zorder=-1)
    #if(plot_detail):
        #if(select_matrix==0):
            #plt.suptitle(f'{w_1strow}')
        #plt.title(f'period:{P},EPPs:{cnt_EPP}')
    
    #全ての点をプロット
    if(plot_allpoint):
        for i in range(2**N):
            plt.scatter(xdec[i]/(2**N),ydec[i]/(2**N),color ='black',s=10)
            
    #初期値を含む軌道
    if(plot_allorbit):
        for i in range(2**N-1):
            plt.plot((xline[i]/(2**N),yline[i]/(2**N)),(yline[i]/(2**N),yline[i]/(2**N)),color='blue',linewidth=1)
            plt.plot((yline[i]/(2**N),yline[i]/(2**N)),(xline[i+1]/(2**N),yline[i+1]/(2**N)),color='blue',linewidth=1)
    
    #BPO
    if(plot_BPO):
        for i in range(P):
            if(i<P-1):
                plt.plot((BPPx[i]/(2**N),BPPy[i]/(2**N)),(BPPy[i]/(2**N),BPPy[i]/(2**N)),color='red',linewidth=1)
                plt.plot((BPPy[i]/(2**N),BPPy[i]/(2**N)),(BPPx[i+1]/(2**N),BPPy[i+1]/(2**N)),color='red',linewidth=1)
            elif(i==P-1):
                plt.plot((BPPx[i]/(2**N),BPPy[i]/(2**N)),(BPPy[i]/(2**N),BPPy[i]/(2**N)),color='red',linewidth=1)
                plt.plot((BPPy[i]/(2**N),BPPy[i]/(2**N)),(BPPx[0]/(2**N),BPPy[0]/(2**N)),color='red',linewidth=1)
    
    #BPO_ohters
    if(plot_BPO_others):
        if any(any(element != 0 for element in row) for row in BPPx_others):
            for i in range(cnt_BPO_others):                
                for j in range(int(length_BPPx2[i])):
                    if(j<int(length_BPPx2[i])-1):
                        plt.plot((BPPx_others[i][j]/(2**N),BPPx_others[i][j+1]/(2**N)),(BPPx_others[i][j+1]/(2**N),BPPx_others[i][j+1]/(2**N)),color='black',linewidth=1.5,alpha=0.2)
                        plt.plot((BPPx_others[i][j+1]/(2**N),BPPx_others[i][j+1]/(2**N)),(BPPx_others[i][j+1]/(2**N),BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N)),color='black',linewidth=1.5,alpha=0.2)
                        plt.plot((BPPx_others[i][j+1]/(2**N),BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N)),(BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N),BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N)),color='black',linewidth=1.5,alpha=0.2)
                        plt.plot((BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N),BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N)),(BPPx_others[i][int((j+2)%length_BPPx2[i])]/(2**N),BPPx_others[i][int((j+3)%length_BPPx2[i])]/(2**N)),color='black',linewidth=1.5,alpha=0.2)
            
    
    if(plot_BPP):
        for i in range(P):
            plt.scatter(BPPx[i]/(2**N),BPPy[i]/(2**N),color='red',s=10,zorder=3)
    
    #EPP
    if(plot_EPP):
        for i in range(len(EPPx)):
            plt.scatter(EPPx[i]/(2**N),EPPy[i]/(2**N),color ='blue',s=10,zorder=2)
            #plt.scatter(EPPx[i]/(2**N),EPPy[i]/(2**N),color ='white',s=3,zorder=2)
    
    #LSP
    if(plot_LSP):
        for i in range(len(LSP)):
            plt.scatter(LSP[i]/(2**N),ydec[int(LSP[i])]/(2**N),color ='blue',s=10,zorder=2)
            
    #DSP
    if(plot_DSP):
        for i in range(len(DSP)):
            plt.scatter(DSP[i]/(2**N),ydec[int(DSP[i])]/(2**N),color ='blue',s=10)
            plt.scatter(DSP[i]/(2**N),ydec[int(DSP[i])]/(2**N),color ='white',s=2,zorder=2)
        
    #GSP
    if(plot_GSP):
        for i in range(len(GSP)):
            plt.scatter(GSP[i]/(2**N),ydec[int(GSP[i])]/(2**N),color ='blue',s=10)
    
    #EPP for fixed point
    if(plot_EPP_FP):
        for i in range(len(fall_fixedpoint)):
            plt.scatter(fall_fixedpoint[i]/(2**N),ydec[int(fall_fixedpoint[i])]/(2**N),color ='blue',s=10)
    
    #DEPP for fixed point
    if(plot_DEPP_FP):
        for i in range(len(DEPP_fixedpoint)):
            plt.scatter(DEPP_fixedpoint[i]/(2**N),ydec[int(DEPP_fixedpoint[i])]/(2**N),color ='blue',s=10)
            plt.scatter(DEPP_fixedpoint[i]/(2**N),ydec[int(DEPP_fixedpoint[i])]/(2**N),color ='white',s=2)
    
    #fixed point
    if(plot_FP):
        for i in range(len(fixedpoint)):
            plt.scatter(fixedpoint[i]/(2**N),ydec[int(fixedpoint[i])]/(2**N),color ='red',s=10)
        
    plt.show()




#時空パターンを計算
xb=np.zeros((horizontal*N,N,4))#ブロック一つ分のx座標
yb=np.zeros((horizontal*N,N,4))#ブロック一つ分のy座標
block=np.zeros((2**N,N))#それぞれの時間における-1,1の値
block_shift=np.zeros((2**N,N))#シフトを計算するため

for i in range(t_start, min(2**N, horizontal + t_start + 1)):
    temp_xline=xline[i]
    for j in range(2**N):
        if(yline[i]==xdec[j]):
            #2進数に変換
            for k in range(N):
                if(temp_xline%2==0):
                    block[i][N-k-1]=-1
                    temp_xline=temp_xline//2
                else:
                    block[i][N-k-1]=1
                    temp_xline=temp_xline//2
                    
            #nビットのブロックを出力
            idx = i - t_start
            if(idx >= 0 and idx <= horizontal):
                for k in range(N):
                    if(block[i][k]==1):#n-k-1を[k]にすると野中さんと同じ,縦軸の1が最大桁になる
                        xb[idx][k] = [idx-0.5, idx+0.5, idx+0.5, idx-0.5]
                        yb[idx][k] = [k+0.5, k+0.5, k+1.5, k+1.5]
                block_shift[horizontal-i]=block[i]#過渡現象を避けるため逆向きに代入している

if(print_block):
    print('時空パターンのバイナリー表示')
    print(block[0:13])#0~12を表示13は含まれない





#時空パターンを描画
if(plot_pattern):
    #グラフの体裁
    plt.figure(dpi=300,figsize=(int(4*horizontal/N),int(4.5*N/N)))
    plt.xlim(-1.7/N*6,(horizontal+1)/N*6)
    plt.ylim(-0.5/N*6,(N+1.5)/N*6)
    plt.axis('off')
    
    plt.axis('off')
    
    if(plot_skelton):
        # 図全体の背景は透明
        plt.gcf().patch.set_alpha(0)
        plt.gca().patch.set_alpha(0)
    
    # ★ セルが並んでいる部分だけ白で塗りつぶす
    cell_xmin = (-0.5) / N * 6
    cell_xmax = (horizontal + 0.5) / N * 6
    cell_ymin = 0.5 / N * 6
    cell_ymax = (N + 0.5) / N * 6
    
    plt.gca().add_patch(
        plt.Rectangle(
            (cell_xmin, cell_ymin),           # 左下
            cell_xmax - cell_xmin,            # 幅
            cell_ymax - cell_ymin,            # 高さ
            facecolor='white',
            edgecolor='none',
            zorder=-10                        # 一番下に
        )
    )

    
    for i in range(N+1):#横線を描画
        if(i==0 or i==N):
            plt.plot(((horizontal+0.5)/N*6,-0.5/N*6),((i+0.5)/N*6,(i+0.5)/N*6),color='black',linewidth=1.5/N*6,zorder=3)
        else:
            plt.plot(((horizontal+0.5)/N*6,-0.5/N*6),((i+0.5)/N*6,(i+0.5)/N*6),color='gray',linewidth=1.5/N*6,zorder=2)
    for i in range(horizontal+2):#縦線を描画
        if(i==0 or i==horizontal+1):
            plt.plot(((i-0.5)/N*6,(i-0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='black',linewidth=1.5/N*6,zorder=3)
        else:
            plt.plot(((i-0.5)/N*6,(i-0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='gray',linewidth=1.5/N*6,zorder=2)
    #for i in range(horizontal):
        #plt.plot((i/N*6,i/N*6),(0.25/N*6,0.45/N*6),color='black',linewidth=1)
    for t in range(t_start, t_start + horizontal + 1):
        if t % 5 == 0: 
            i = t - t_start
            
            plt.plot((i/N*6, i/N*6), (0.2/N*6, 0.45/N*6),color='black', linewidth=2)
            plt.text(i/N*6, -0.5/N*6, f"{t}",ha="center", size=20/N*6,fontproperties=cambriamath_font)
    
    if(plot_idx):
        for i in range(N):
            #plt.plot((-0.7/N*6,-0.5/N*6),((i+1)/N*6,(i+1)/N*6),color='black',linewidth=1)
            plt.text(-0.9/N*6,(i+1)/N*6,f"{int(i+1)}",ha="center",va="center",size=20/N*6,fontproperties=cambriamath_font)
    
    if(plot_blackspace):
        for i in range(horizontal+1):
            for j in range(N):
                plt.fill(xb[i][j]/N*6,yb[i][j]/N*6,color='black')

    if(plot_xout_before):
        for i in range(horizontal):
            for j in range(N):
                if(xout_before[int(xline[i])][0][j]>0):
                    plt.text((i+1)/N*6,(j+1)/N*6,f"+{int(xout_before[int(xline[i])][0][j])}",color="gray",fontsize=17/N*6,fontproperties=cambriamath_font,ha="center",va="center")
                else:
                    plt.text((i+1)/N*6,(j+1)/N*6,f"{int(xout_before[int(xline[i])][0][j])}",color="gray",fontsize=17/N*6,fontproperties=cambriamath_font,ha="center",va="center")

    if(plot_xout_after):
        for i in range(N):
            if(xin[InitialValue][0][i]>0):
                plt.text(0,(i+1)/N*6,f"+{int(xin[InitialValue][0][i])}",color="gray",fontsize=17/N*6,fontproperties=cambriamath_font,ha="center",va="center")
            else:
                plt.text(0,(i+1)/N*6,f"{int(xin[InitialValue][0][i])}",color="gray",fontsize=17/N*6,fontproperties=cambriamath_font,ha="center",va="center")
        for i in range(horizontal):
            for j in range(N):
                if(xout[int(xline[i])][0][j]>0):
                    plt.text((i+1)/N*6,(j+1)/N*6,f"+{int(xout[int(xline[i])][0][j])}",color="gray",fontsize=17/N*6,fontproperties=cambriamath_font,ha="center",va="center")
                else:
                    plt.text((i+1)/N*6,(j+1)/N*6,f"{int(xout[int(xline[i])][0][j])}",color="gray",fontsize=17/N*6,fontproperties=cambriamath_font,ha="center",va="center")
                
    if(plot_redframe):
        edge_left_global  = np.min(np.where(xline == int(BPPx[0]))[0])
        edge_right_global = np.min(np.where(xline == int(BPPx[int(P-1)]))[0])
        edge_left  = edge_left_global  - t_start
        edge_right = edge_right_global - t_start
        
        if(0 <= edge_left <= horizontal):
            plt.plot(((edge_left-0.5-0.09)/N*6,(edge_left-0.5-0.09)/N*6),((N+0.5)/N*6,0.5/N*6),color='red',linewidth=4/N*6,zorder=4)
        
        if(0 <= edge_right <= horizontal):
            plt.plot(((edge_right+0.5+0.09)/N*6,(edge_right+0.5+0.09)/N*6),((N+0.5)/N*6,0.5/N*6),color='red',linewidth=4/N*6,zorder=4)
            
        left_draw  = max(edge_left, 0)
        right_draw = min(edge_right, horizontal)
        
        if(right_draw >= left_draw):
            # 上下の横線（見える範囲にクリップ）
            if(edge_left < 0):
                left_vis = -0.5
            else:
                left_vis = edge_left - 0.5 - 0.09
            
            if(edge_right > horizontal):
                right_vis = horizontal + 0.5
            else:
                right_vis = edge_right + 0.5 + 0.09
                
            # 上の横線
            plt.plot((left_vis/N*6, right_vis/N*6), ((N+0.5+0.09)/N*6, (N+0.5+0.09)/N*6),color='red', linewidth=4/N*6, zorder=4)
            
            # 下の横線
            plt.plot((left_vis/N*6, right_vis/N*6),((0.5-0.09)/N*6, (0.5-0.09)/N*6),color='red', linewidth=4/N*6, zorder=4)

            
    for x in plot_frame_blue:
        plt.plot(((x-0.5)/N*6,(x-0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='royalblue',linewidth=6/N*6,zorder=4)
        plt.plot(((x+0.5)/N*6,(x+0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='royalblue',linewidth=6/N*6,zorder=4)
        plt.plot(((x-0.5)/N*6,(x+0.5)/N*6),(0.5/N*6,0.5/N*6),color='royalblue',linewidth=6/N*6,zorder=4)
        plt.plot(((x-0.5)/N*6,(x+0.5)/N*6),((N+0.5)/N*6,(N+0.5)/N*6),color='royalblue',linewidth=6/N*6,zorder=4)
    for x in plot_frame_green:
        plt.plot(((x-0.5)/N*6,(x-0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='forestgreen',linewidth=6/N*6,zorder=4)
        plt.plot(((x+0.5)/N*6,(x+0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='forestgreen',linewidth=6/N*6,zorder=4)
        plt.plot(((x-0.5)/N*6,(x+0.5)/N*6),(0.5/N*6,0.5/N*6),color='forestgreen',linewidth=6/N*6,zorder=4)
        plt.plot(((x-0.5)/N*6,(x+0.5)/N*6),((N+0.5)/N*6,(N+0.5)/N*6),color='forestgreen',linewidth=6/N*6,zorder=4)
    for x in plot_frame_orange:
        plt.plot(((x-0.5)/N*6,(x-0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='darkorange',linewidth=6/N*6,zorder=4)
        plt.plot(((x+0.5)/N*6,(x+0.5)/N*6),((N+0.5)/N*6,0.5/N*6),color='darkorange',linewidth=6/N*6,zorder=4)
        plt.plot(((x-0.5)/N*6,(x+0.5)/N*6),(0.5/N*6,0.5/N*6),color='darkorange',linewidth=6/N*6,zorder=4)
        plt.plot(((x-0.5)/N*6,(x+0.5)/N*6),((N+0.5)/N*6,(N+0.5)/N*6),color='darkorange',linewidth=6/N*6,zorder=4)
    
    
    if(plot_frame_red2):
        plt.plot(((plot_frame_red2[0]-0.45)/N*6,(plot_frame_red2[0]-0.45)/N*6),((N+0.45)/N*6,0.55/N*6),color='red',linewidth=3/N*6,zorder=4)
        plt.plot(((plot_frame_red2[1]+0.45)/N*6,(plot_frame_red2[1]+0.45)/N*6),((N+0.45)/N*6,0.55/N*6),color='red',linewidth=3/N*6,zorder=4)
        plt.plot(((plot_frame_red2[0]-0.45)/N*6,(plot_frame_red2[1]+0.45)/N*6),(0.55/N*6,0.55/N*6),color='red',linewidth=3/N*6,zorder=4)
        plt.plot(((plot_frame_red2[0]-0.45)/N*6,(plot_frame_red2[1]+0.45)/N*6),((N+0.45)/N*6,(N+0.45)/N*6),color='red',linewidth=3/N*6,zorder=4)
    if(plot_frame_blue2):
        plt.plot(((plot_frame_blue2[0]-0.45)/N*6,(plot_frame_blue2[0]-0.45)/N*6),((N+0.45)/N*6,0.55/N*6),color='royalblue',linewidth=3/N*6,zorder=4)
        plt.plot(((plot_frame_blue2[1]+0.45)/N*6,(plot_frame_blue2[1]+0.45)/N*6),((N+0.45)/N*6,0.55/N*6),color='royalblue',linewidth=3/N*6,zorder=4)
        plt.plot(((plot_frame_blue2[0]-0.45)/N*6,(plot_frame_blue2[1]+0.45)/N*6),(0.55/N*6,0.55/N*6),color='royalblue',linewidth=3/N*6,zorder=4)
        plt.plot(((plot_frame_blue2[0]-0.45)/N*6,(plot_frame_blue2[1]+0.45)/N*6),((N+0.45)/N*6,(N+0.45)/N*6),color='royalblue',linewidth=3/N*6,zorder=4)

    plt.show()
if(plot_energy):
    if(model==0):
        if(layer==2):
            plt.figure(dpi=300, figsize=(4, 4))
            for i in range(2**N):
                plt.scatter(i,E[i],color='black',s=10)
                if (i in BPPx):
                   plt.scatter(i,E[i],color='red',s=10) 
                if (i in EPPx):
                   plt.scatter(i,E[i],color='blue',s=10) 
            plt.show()

#プリント
#色付きでプリントする関数を定義
def printcolored(array):
    for row in array:
        for element in row:
            if element<0:
                print(Style.BRIGHT+Fore.BLUE+str(int(element)),end=" ")  # 明るい青色
            elif element==0:
                print(end=' ')
                print(Style.BRIGHT+Fore.WHITE+str(int(element)),end=" ")  # 明るい白色
            elif element>0:
                print(Style.BRIGHT+Fore.RED+f"+{int(element)}",end=" ")  # 明るい赤色
            else:
                print(str(element),end=" ")
        print()
    print(Style.RESET_ALL)  #スタイルをリセット

if(model==0):
    if(print_conection_matrix):
        printcolored(w)
        if(layer==3):
            printcolored(w_L2)

    if(print_cnt_input):
        if(select_matrix==0):
            print(f"入力数:{cnt_input}")
            if(layer==3):
                if(select_matrix_L2==0):
                    print(f"入力数:{cnt_input_L2}")
        elif(select_matrix==1):
            print(f"入力数(1~{int(N/2)}行目):{cnt_input1}")
            print(f"入力数({int(N/2+1)}~{N}行目)]{cnt_input4}")
elif(model==1):
    if(select_CA==0):
        print(f"RuleNumber:{RN}")

if(print_cnt_BPO):
    print(f"BPOの数:{cnt_BPO}個")
    
if(print_longest_period):
    if(longest_period==1):
        print(f"最長周期:周期{longest_period}(不動点)")
    elif(longest_period>1):
        print(f"最長周期:周期{longest_period}")

if(print_longest_period_EPP):
    #if(BPPx.size>1):
    print(f"最長周期のEPP:{int(cnt_longest_period_EPP)}個")

if(print_most_EPP):
    print(f"最多EPP:{int(most_EPP)}個")

if(print_most_EPP_period):
    print(f"最多EPPの周期:周期{int(most_EPP_period)}")

if(print_fixedpoint):
    print(f"不動点:{cnt_fixedpoint}個")
    if(cnt_fixedpoint!=0):
        print([int(x) for x in fixedpoint])

if(print_fall_fixedpoint):
    print(f"EFP:{len(fall_fixedpoint)}個")
    if(len(fall_fixedpoint)!=0):
        print(f"{fall_fixedpoint}")
        print(f"DEFP:{len(DEPP_fixedpoint)}個")

if(print_step_IV):
    print("収束するまでのステップ数:")
    print([int(x) for x in step_IV])

if(model==0):
    if(print_eig_val):
        print("固有値:")
        print(eig_val)
        if(layer==3):
            print(eig_val_L2)

print("\n")

if(print_orbit):
    if(P>1):
        print(f"初期値{InitialValue}の周期軌道")
        print([int(x) for x in BPPx])
        print(f"周期{P}")
    else:
        print(f"初期値{InitialValue}は不動点{int(BPPx[0])}に収束")

if(print_cnt_EPP):
    if(P>1):
        print(f"EPP:{cnt_EPP}/{notBPP}個")
    else:
        print(f"不動点{int(BPPx[0])}に落ち込む初期値:{len(fall_fixedpoint_initial)}個")
        if(len(fall_fixedpoint_initial)!=0):
            print([int(x) for x in fall_fixedpoint_initial])
if(print_EPP):
    if(len(EPPx)!=0):
        formatted_EPPx=[f"{int(x)}({'→' * int(step_IV[int(x)])}{int(BPPfromIV[int(x)])})" for x in EPPx]
        print(f"[{', '.join(formatted_EPPx)}]")
if(print_cnt_LSP):
    if(P>1):
        print(f"L安定:{len(LSP)}/{len(BitError_dec)}個")
        #HummingDistanceが1の点をClosestNeighborという．
if(print_LSP):
    if(P>1):
        if(len(LSP)!=0):
            formatted_LSP=[f"{int(x)}({'→' * int(step_IV[int(x)])}{int(BPPfromIV[int(x)])})" for x in LSP]
            print(f"[{', '.join(formatted_LSP)}]")
if(print_BitError):
    if(P>1):
        print("1ビットエラー:")
        print([int(x) for x in BitError_dec])
if(print_cnt_DSP):
    if(P>1):
        print(f"D安定:{len(DSP)}個")
        #DEPPともいう．
if(print_DSP):
    if(len(DSP)!=0):
        print([int(x) for x in DSP])
if(print_cnt_GSP):
    if(P>1):
        print(f"G安定:{len(GSP)}個")
if(print_GSP):
    if(len(GSP)!=0):
        print([int(x) for x in GSP])

print("\n")

if(print_orbit_others):
    print("その他の周期軌道")
    if all(all(element == 0 for element in row) for row in BPPx_others):
        print("なし")
    else:
        for i in range(cnt_BPO_others):
            print([int(x) for x in BPPx_others[i][0:int(length_BPPx2[i])]])
            print(f"周期{int(length_BPPx2[i])}")
            print(f"EPP:{int(EPPx_others[i])}個")
            print()
if(print_fixedpoint_others):
    print("その他の不動点")
    if (np.size(fixedpoint_others)==0):
        print("なし")
    else:
        for i in range(len(fixedpoint_others)):
            print([int(fixedpoint_others[i])])
            print(f"EFP:{int(fall_fixedpoint_others[i])}個")
            print()

time_end=time.time() #処理時間のおわり
time_diff=time_end-time_start
time_end_formatted=time.strftime("%Y/%m/%d %H:%M:%S",time.localtime(time_end))
time_diff_formatted=time.strftime("%H:%M:%S",time.gmtime(time_diff))

if(print_runtime):
    print()
    print(f"実行終了:{time_end_formatted}")
    print(f"処理時間:{time_diff_formatted}")