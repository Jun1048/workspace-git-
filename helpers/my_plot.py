import os

from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from pandas import pivot_table
from scipy.spatial import ConvexHull
from . import my_stats
import seaborn as sb
import numpy as np
import glob as gl


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 그래프 기본 설정
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
my_dpi = 200                                         # 이미지 선명도(100~300)
plt.rcParams['font.size'] = 12                      # 기본 폰트 크기
plt.rcParams['axes.unicode_minus'] = False          # 그래프에 마이너스 깨짐 방지
plt.rcParams['figure.dpi'] = my_dpi                 # 그래프의 dpi 설정
plt.rcParams['savefig.dpi'] = my_dpi                # 저장되는 그래프의 dpi 설정
plt.rcParams['lines.linewidth'] = 2                 # 그래프 선 굵기 설정
plt.rcParams['axes.axisbelow'] = True               # 그래프의 축과 격자선을 뒤에 배치



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 그래프 초기화 함수 정의하기
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def init(width=1280, height=640, rows=1, cols=1, title=None, xlabel=None,
             ylabel=None, grid=True, twinx=False):
    
    my_figsize = ((width / 100) * cols, (height / 100) * rows)

    fig, ax = plt.subplots(rows, cols, figsize=my_figsize, dpi=200)

    if rows > 1  or cols > 1:
        ax = ax.flatten()          # 2차원 배열을 1차원으로 평탄화 하여 반복 처리
        fig.suptitle(title, fontsize=32, fontweight=500)
        
        for a in ax:
            a.grid(grid, alpha=0.5)
    
    else:
        ax.grid(grid, alpha=0.5)
    
        if title:
            ax.set_title(title, fontsize=24, fontweight=500, pad=15)

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=16, fontweight=400, labelpad=5)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=16, fontweight=400, labelpad=5)
    
    if twinx : 
        ax_right = ax.twinx()
        ax = (ax, ax_right)
        
    return fig, ax


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 그래프 출력 함수 정의하기
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def show(save_path=None):
    if save_path:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 시계열 그래프 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lineplot(data=None, x=None, y=None, hue=None,
             title=None, xlabel=None, ylabel=None,
             color=None, linewidth=2.0, linestyle="-", palette=None,
             marker=None, markersize=None, markeredgewidth=None,
             markeredgecolor=None, markerfacecolor=None,
             width=1280, height=640, save_path=None, ax=None):
    

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 그래프 초기화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fig = None

    if ax is None:
        fig, ax = init(width = width, height = height, title = title, xlabel = xlabel,
                       ylabel = ylabel)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 선 그래프 그리기
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    sb.lineplot(data=data, x=x, y=y, hue=hue,
    color=color, linewidth=linewidth, linestyle=linestyle,
    palette=palette, marker=marker, markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor=markeredgecolor,
    markerfacecolor=markerfacecolor,
    ax=ax)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 그래프 표시
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if fig is not None:
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 다변량 커널밀도 그래프 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def kdeplot(data = None, x=None, hue=None, meanline=False, clevel=0,
            title = None, xlabel = None, ylabel = None,
            fill = False, linewidth = 2.0, palette = None,
            width = 1280, height = 640, save_path = None, ax=None):
    
    # 그래프 초기화
    fig = None
    
    if ax is None:
        fig, ax = init(width=width, height=height, title=title,
                   xlabel=xlabel, ylabel=ylabel)
    
    # 다변량 커널 밀도 그래프 그리기
    sb.kdeplot(data=data, x=x, hue=hue, fill=fill, linewidth=linewidth,
               palette=palette, ax=ax)
    

    # 신뢰구간 표시 (신뢰수준이 0이 아닌 경우에만)
    if clevel:
        ymin, ymax = ax.get_ylim()      # 그래프의 y축 범위 조회

        if hue is None:

            # 그래프에 적용된 팔레트의 첫 번째 색상을 따른다. (팔레트가 없으면 기본 파란색)
            color = sb.color_palette(palette)[0] if palette else '#0066ff'

            # 전체 데이터에 대한 신뢰구간 표시 (보관 함수 호출)
            _draw_ci(ax, my_stats.ci(data, column=x, clevel=clevel), color, ymax)
        
        else:
            # hue 범주별로 신뢰구간 표시(kdeplot이 그린 라인의 색상과 일치시킴)
            categories = list(data[hue].unique())

            # 팔레트에서 범주의 수에 맞는 색상값 추출
            colors = sb.color_palette(palette, n_colors=len(categories))

            # 각 범주에 대해 신뢰구간 표시
            for i, cat in enumerate(categories):
                cdata = data.loc[data[hue] == cat, x]
                _draw_ci(ax, my_stats.ci(cdata, clevel=clevel), colors[i], ymax)
        
        ax.set_ylim(ymin, ymax)     # y축 범위 유지


    # 평균선 표시
    if meanline:
        y_max = ax.get_ylim()[1]

        if hue is None:
            mv = data[x].mean()
            ax.axvline(x=mv, color='red', linestyle='--', linewidth=linewidth * 0.5)
            ax.text(x=mv + 0.05, y=y_max * 0.95, s=f'Mean: {mv:.2f}', color='red', fontsize=14,
                    ha='center')
        else:
            # hue 범주별 평균선 표시 (kdeplot이 그린 라인의 색상과 일치시킴)
            categories = list(data[hue].unique())

            # 팔레트에서 범주의 수에 맞는 색상값 추출
            colors = sb.color_palette(palette, n_colors=len(categories))

            # 각 범주에 대해 평균선 표시
            for i, cat in enumerate(categories):
                 mv = data.loc[data[hue] == cat, x].mean()
                 ax.axvline(x=mv, color=colors[i], linestyle='--', linewidth=linewidth * 0.5)
                 ax.text(x=mv + 0.05, y=y_max * (0.95 - i * 0.07), s=f'{cat} Mean: {mv:.2f}',
                         color=colors[i], fontsize=14, ha='center')
    
    # 그래프 표시
    if fig is not None:
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 히스토그램 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def histplot(data=None, x=None, hue=None, bins="auto",
             title=None, xlabel=None, ylabel=None,
             linewidth=1, palette=None, kde=False,
             width=1280, height=640, save_path=None, ax=None):
    
    # 그래프 초기화
    fig = None
    
    if ax is None:
        fig, ax = init(width=width, height=height, title=title,
                       xlabel=xlabel, ylabel=ylabel)
    
    # 구간 산정
    if isinstance(bins, int):
        hist, bins = np.histogram(data[x], bins=bins)
        bins = np.round(bins, 1)
        ax.set_xticks(bins, bins)
    elif isinstance(bins, (list, np.ndarray)):
        ax.set_xticks(bins, bins)

    # 히스토그램 그리기
    sb.histplot(data=data, x=x, hue=hue, linewidth=linewidth,
                palette=palette, kde=kde, bins=bins, ax=ax)
    
    # 그래프 표시
    if fig is not None:
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 박스 플롯 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def boxplot(data=None, x=None, y=None, hue=None, orient=None,
            palette=None, order=None, title=None, xlabel=None, ylabel=None,
            width=1280, height=640, save_path=None, ax=None, legend=False):
    
    # 그래프 초기화
    fig = None
    
    if ax is None:
        fig, ax = init(width=width, height=height, title=title,
                   xlabel=xlabel, ylabel=ylabel)
    

    # 박스 플롯 그리기
    sb.boxplot(data=data, x=x, y=y, hue=hue, orient=orient,
               order=order, palette=palette, ax=ax, legend=legend)
    
    
    # 그래프 표시
    if fig is not None:
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 바이올린 플롯 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def violinplot(data=None, x=None, y=None, hue=None, orient=None,
            palette=None, title=None, xlabel=None, ylabel=None,
            width=1280, height=640, save_path=None, ax=None):
    
    # 그래프 초기화
    fig = None

    if ax is None:
        fig, ax = init(width=width, height=height, title=title, xlabel=xlabel,
                       ylabel=ylabel)
    

    # 바이올린 플롯 그리기
    sb.violinplot(data=data, x=x, y=y, hue=hue, orient=orient,
                  palette=palette, ax=ax)
    
    
    # 그래프 표시
    if fig is not None:
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 히트맵 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def heatmap(data=None, annot=True, fmt="0.2f", linewidth=0.5,
            palette=None, title=None, xlabel=None, ylabel=None,
            width=1280, height=640, save_path=None, ax=None):
    
    # 그래프 초기화
    fig = None

    if ax is None:    
        fig, ax = init(width=width, height=height, title=title, xlabel=xlabel, ylabel=ylabel)


    # 그리드 제거
    ax.grid(False)


    # 히트맵 그리기
    sb.heatmap(data=data, annot=annot, fmt=fmt, linewidth=linewidth,
               cmap=palette, ax=ax)
    
    
    # 그래프 표시
    if fig is not None:    
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bar plot 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def barplot(data=None, x=None, y=None, hue=None, estimator=np.mean,
            order=None, palette=None, title=None, xlabel=None, ylabel=None,
            width=1280, height=640, save_path=None, ax=None):
    
    # 그래프 초기화
    fig = None

    if ax is None:     
        fig, ax = init(width=width, height=height,
                       title=title, xlabel=xlabel, ylabel=ylabel)


    # Bar plot 그리기
    sb.barplot(data=data, x=x, y=y, hue=hue, estimator=estimator,
               order=order, palette=palette, ax=ax, legend=True)
    
    ax.legend()
    
    # 그래프 표시
    if fig is not None:    
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Count plot 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def countplot(data=None, x=None, y=None, hue=None, order=None,
              palette=None, title=None, xlabel=None, ylabel=None,
              width=1280, height=640, save_path=None, ax=None, dodge=True, legend=True):
    
    # 그래프 초기화
    fig = None

    if ax is None:      
        fig, ax = init(width=width, height=height,
                   title=title, xlabel=xlabel, ylabel=ylabel)


    # Count plot 그리기
    sb.countplot(data=data, x=x, y=y, hue=hue, order=order,
                 palette=palette, ax=ax, dodge=dodge, legend=legend)
    
    
    # 그래프 표시
    if fig is not None: 
        show(save_path=save_path)




# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 파이/도넛 차트 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pieplot(x, labels, autopct="%0.1f%%", startangle=90, counterclock=False,
            explode=None, donutchart=False,
            wedge_width=0.7, wedge_color="#ffffff", wedge_linewidth=3,
            palette=None, title=None, xlabel=None, ylabel=None,
            width=1280, height=640, save_path=None, ax=None):
    
    
    # 그래프 초기화
    fig = None

    if ax is None:     
        fig, ax = init(width=width, height=height, title=title,
                       xlabel=xlabel, ylabel=ylabel)
    

    # 색상값을 팔레트로부터 추출
    color_list = None
    if palette:
        color_list = sb.color_palette(palette, n_colors=len(labels))

    
    # 도넛 그래프 그리기 옵션 생성
    wedgeprops = None
    if donutchart:
        wedgeprops={"width": wedge_width, "edgecolor": wedge_color,
                    "linewidth": wedge_linewidth}
        
        
    # 파이 그래프 그리기
    ax.pie(x, labels=labels, autopct=autopct, startangle=startangle,
           counterclock=counterclock, explode=explode,
           colors=color_list, wedgeprops=wedgeprops)
    

    # 그래프 표시
    if fig is not None:     
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Stacked Bar 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stackplot(data, x, y, hue, aggfunc=np.sum, orient='v', ratio=False,
              text=True, text_color="#ffffff", text_fontsize=12,
              text_format="{:.1f}",
              palette=None, title=None, xlabel=None, ylabel=None,
              width=1280, height=640, save_path=None, ax=None):
    
    
    # 그래프 초기화
    fig = None

    if ax is None:      
        fig, ax = init(width=width, height=height, title=title,
                       xlabel=xlabel, ylabel=ylabel)
    

    # 데이터 피벗팅 (fill_value=0 → 결측치를 0으로 채움)후 인덱스를 문자열 카테고리로 변환
    df = pivot_table(data=data, index=x, values=y, columns=hue, aggfunc=aggfunc,
                     fill_value=0)
    df.index = df.index.astype("str").astype("category")

    
      # 누적값을 비율로 변환하는 경우
    if ratio : 
        if text_format is None:
            text_format = '{:.1f}%'

        df['sum'] = df.sum(axis=1)

        for col in df.columns:
            df[col] = df[col] / df['sum'] * 100

        df.drop(columns='sum', inplace=True)

        if orient =='v':
            ax.set_ylim(0, 100)

        else : 
            ax.set_xlim(0, 100)
    else:
        if text_format is None:
            text_format = '{:.1f}'



    # 색상값 생성하기
    color_list = None
    if palette is not None:
        color_list = sb.color_palette(palette, n_colors=len(df.columns))
        
        
    # 피벗테이블의 각 열에 대해 누적 막대그래프 그리기
    for i, col in enumerate(df.columns):
        color = None

        if color_list is not None:
            color = color_list[i]

    # 세로 그래프
        if orient == 'v':
            ax.bar(df.index, df[col], bottom=df.iloc[:, :i].sum(axis=1), color=color,
            label=col)
        
    # 가로 그래프    
        else:
            ax.barh(df.index, df[col], left=df.iloc[:, :i].sum(axis=1), color=color,
            label=col)
        
    # 누적값 테스트 표시
        if text:
            for j, val in enumerate(df[col]):
                if val == 0: # 누적값이 0인 경우 텍스트 표시하지 않음
                    continue

                if orient == 'v':
                    ax.text(x=j, y=df.iloc[j, :i].sum() + val / 2,
                        s=text_format.format(val), ha='center', va='center',
                        color=text_color, fontsize=text_fontsize)
            
                else:
                    ax.text(x=df.iloc[j, :i].sum() + val / 2, y=j,
                        s=text_format.format(val), ha='center', va='center',
                        color=text_color, fontsize=text_fontsize)
            

    # 범례 표시
    ax.legend(bbox_to_anchor=(1,1))


    # 그래프 표시
    if fig is not None:       
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Convex Hull
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_hull(data, x, y, hue, palette, ax):


    # 데이터의 군집 종류 얻기
    classes = list(data[hue].unique())


    # 각 클래스에 대하여 반복 수행
    for i, v in enumerate(classes):


        # 현재 클래스에 해당하는 데이터 포인트 추출
        df_c = data.loc[data[hue] == v, [x, y]]


        # ConvexHull은 3개 이상의 점이 필요하므로, 데이터 포인트가 3개 미만인 경우 중단해야 함
        if len(df_c) < 3:
            continue
    
        hull = ConvexHull(df_c)
        points = np.append(hull.vertices, hull.vertices[0])

        # 현재 클래스에 적용될 색상값 생성
        color = sb.color_palette(palette)[i]

        # points를 index로 하는 데이터 포인트를 선과 면으로 표시
        ax.plot(df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":", color=color)
        ax.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1, color=color)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 산점도 그래프 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def scatterplot(data, x, y, hue=None, marker='o', color=None, size=100,
                edgecolor="#ffffff", linewidth=1.5, alpha=1, palette='tab10',
                outline=True, title=None, xlabel=None, ylabel=None,
                width=1280, height=640, save_path=None, ax=None):
    

    # 그래프 초기화
    fig = None

    if ax is None:
        fig, ax = init(width=width, height=height, title=title,
                       xlabel=xlabel, ylabel=ylabel)


    # 군집을 구분할 분류값이 없다면 palette 옵션이 무의미하므로 None으로 설정
    if hue == None:
        if color is None and palette is not None:
            color = sb.color_palette(palette)[0]

        palette = None
    else:
        color = None


    # 산점도 그리기
    sb.scatterplot(data=data, x=x, y=y,
                   hue=hue, # 군집을 구분할 분류값이 있는 컬럼명
                   color=color, # 마커 색상
                   palette=palette, # 색상 팔레트 설정
                   marker=marker, # 마커 모양
                   s=size, # 마커 크기 (기본값=100)
                   edgecolor=edgecolor, # 마커 테두리 색상
                   linewidth=linewidth, # 마커 테두리 두께
                   alpha=alpha,
                   ax=ax) # 마커 투명도
    

    # 외곽선 그리기
    if outline and hue is not None:
        
        # 외곽선 그리기
        plot_hull(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)

    # 그래프 표시
    if fig is not None:
        show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# lm plot 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lmplot(data, x, y, hue=None, palette=None, col=None, row=None, markers="o",
           scatter_edgecolor="#ffffff", scatter_linewidths=1, scatter_size=50,
           scatter_alpha= 0.8, linestyle="-", linecolor=None, linewidth= 2,
           title= None, xlabel= None, ylabel= None, width=1280, height=640,
           save_path= None):
    
    
    # 1) 그래프 초기화
    w = width / 100 # 가로 크기
    h = height / 100 # 세로 크기
    my_dpi = 200 # 해상도 설정

    # hue가 지정되지 않았는데 palette와 linecolor가 지정된 경우, 무의미하므로 None으로 설정
    if not hue and palette:
        palette = None
        linecolor = None


    # 2) lmplot 그리기
    g = sb.lmplot(data=data, x=x, y=y, height=h, aspect=w/h, hue=hue, col=col, row=row,
                  legend=False, markers=markers, palette=palette,
                  
                  scatter_kws={
                               "edgecolor": scatter_edgecolor, "linewidths": scatter_linewidths,
                               "s": scatter_size, "alpha": scatter_alpha },
                  line_kws={
                            "linestyle": linestyle, "color": linecolor,
                            "linewidth": linewidth }
                    )
    

    # 3) 그래프 설정 및 표시
    g.fig.set_dpi(my_dpi)
    g.fig.set_tight_layout(True)
    ax = g.axes.flatten()

    if title:
        g.fig.suptitle(title, fontsize=24, fontweight=500, y=1)

    for x in g.axes.flatten():
        x.grid(True, alpha=0.5)
        x.set_axisbelow(True)

        if xlabel: x.set_xlabel(xlabel, fontsize=16, fontweight=400, labelpad=5)
        if ylabel: x.set_ylabel(ylabel, fontsize=16, fontweight=400, labelpad=5)

        if hue is not None:
            x.legend(bbox_to_anchor=(1, 1), loc='upper left') # 범례 위치 조정

    show(save_path=save_path)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# pari plot 모듈화
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pairplot(data, x=None, y=None, hue=None, palette=None, diag_kind="kde", reg=False,
             markers="o", scatter_size=20, scatter_alpha=0.8,
             linecolor=None, linewidth=1.5, linestyle="-",
             title=None, width=1280, height=640, save_path=None):
    

    # 1) 그래프 초기화
    figsize = (width / 100, height / 100)

    # hue가 지정되지 않았는데 palette와 linecolor가 지정된 경우, 무의미하므로 None으로 설정
    if not hue and palette:
        palette = None

    # 회귀선의 표시 여부에 따라서 plot_kws 분기
    if reg:
        plot_kws = {
                    "scatter_kws": { "s": scatter_size, "alpha": scatter_alpha},
                    "line_kws": { "color": linecolor, "linewidth": linewidth,
                    "linestyle": linestyle }
                    }
    else:
        plot_kws = { "s": scatter_size, "alpha": scatter_alpha }


    # 2) pairplot 그리기
    g = sb.pairplot(data=data, hue=hue, markers=markers, palette=palette,
                    kind="reg" if reg else "scatter",
                    diag_kind=diag_kind, plot_kws=plot_kws)

    g.fig.set_dpi(200)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])

    if title:
        g.fig.suptitle(title, fontsize=24, fontweight='bold')    
    

    # 3) 개별 그래프 설정 및 화면 출력
    for ax in g.axes.flatten():
        ax.set_axisbelow(True)         # 격자를 그래프 뒤로 이동
        ax.grid(True, alpha=0.5)       # 격자 추가

    show(save_path)                     # 화면 출력




# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 그래프 캔버스 객체(ax)에 신뢰구간을 표시하는 보조함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _draw_ci(ax, interval, color, ymax):
    
    cmin, cmax = interval

    # 신뢰구간 범위에 대한 세로 직선 그리기(cmin, cmax)
    ax.axvline(cmin, linestyle=':', color=color, linewidth=0.5)
    ax.axvline(cmax, linestyle=':', color=color, linewidth=0.5)


    # 신뢰구간 범위에 대한 텍스트 추가
    ax.text(cmin, ymax * 0.9, f'{cmin:.2f}', color=color, fontsize=11, ha='right')
    ax.text(cmax, ymax * 0.9, f'{cmin:.2f}', color=color, fontsize=11, ha='left')

    
    # 신뢰구간 범위에 대한 영역 채우기 (cmin ~ cmax)
    ax.fill_between([cmin, cmax], 0, ymax, alpha=0.1, color=color)




# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 포인트플롯 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pointplot(data=None, x=None, y=None, hue=None, order=None, hue_order=None,
              estimator='mean', errorbar='se', capsize=0.1, dodge=False, markers='o',
              linestyles='-', palette=None, color=None, title=None, xlabel=None, ylabel=None,
              legend_title=None, width=1280, height=640, save_path=None, ax=None):
    

    # 그래프 초기화
    fig = None
    if ax is None:
        fig, ax = init(width=width, height=height, title=title, xlabel=xlabel, ylabel=ylabel)

    
    # 점 그래프 그리기
    sb.pointplot(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order,
              estimator=estimator, errorbar=errorbar, capsize=capsize, dodge=dodge, markers=markers,
              linestyles=linestyles, palette=palette, color=color, ax=ax)
    

    # 범례 제목 설정(hue가 있을 때)
    if hue is not None and legend_title is not None:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(legend_title)

    
    # 그래프 표시
    if fig is not None:
        show(save_path=save_path)