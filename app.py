import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="HouseLens", layout="wide")

def apply_custom_css():
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #2C3E50; }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 5% 5% 5% 10%;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-top: 3px solid #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_merged_data():
    df = pd.read_csv("data/merged_df.csv")
    df['날짜'] = pd.to_datetime(df['년'].astype(str) + '-' + df['월'].astype(str) + '-01')
    return df

@st.cache_data
def load_model_data():
    df = pd.read_csv("data/model_df.csv")
    return df

def configure_sidebar(df):
    st.sidebar.title("분석 조건")
    st.sidebar.markdown("**지역: 서울특별시**")
    min_date = df['날짜'].min().date()
    max_date = df['날짜'].max().date()
    start_date, end_date = st.sidebar.slider("기간 선택", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM")

    filtered_df = df[
        (df['시도'] == '서울특별시') &
        (df['날짜'].dt.date >= start_date) &
        (df['날짜'].dt.date <= end_date)
    ]

    st.sidebar.markdown("---")
    menu = st.sidebar.radio("메뉴", ["Home", "거래량 분석", "경제지표 분석", "회귀분석 결과", "시나리오 분석"])

    return filtered_df, menu

def draw_home(df):
    if df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다. 사이드바에서 필터 조건을 변경해주세요.")
        return

    try:
        latest_date = df['날짜'].max()
        latest_data = df[df['날짜'] == latest_date]

        current_volume = latest_data['거래량'].sum()
        avg_price = latest_data['아파트 매매가격지수'].mean() 
        base_rate = latest_data['기준금리'].mean()
        mortgage_rate = latest_data['주택담보대출금리'].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재 거래량", f"{int(current_volume):,} 건")
        col2.metric("평균 거래금액 (매매지수)", f"{avg_price:,.2f}")
        col3.metric("기준금리", f"{base_rate:,.2f} %")
        col4.metric("주택담보대출 금리", f"{mortgage_rate:,.2f} %")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📌 프로젝트 소개 & 타겟 수요층",
            "📊 데이터셋 & 주요 변수",
            "⚙️ 분석 프레임워크 & 안내",
            "⚠️ 분석 한계점 (Limitations)"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h4 style='color: #2C3E50; margin-bottom: 12px;'>💡 문제 제기 (Background)</h4>
                    <p style='color: #495057; line-height: 1.7;'>
                        <b>아파트 거래량</b>은 부동산 시장의 흐름과 선행 방향성을 가장 빠르게 파악할 수 있는 핵심 지표입니다.<br><br>
                        하지만 <b>기준금리 변화, 주택담보대출금리, 아파트 매매가격지수, 소비자물가지수</b> 등 다양한 거시경제 요인들이 
                        실제 아파트 매매 거래량에 어떻게 복합적인 영향을 미치는지 직관적으로 파악하기는 어렵습니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h4 style='color: #2C3E50; margin-bottom: 12px;'>🎯 프로젝트 목표 (Objectives)</h4>
                    <ul style='color: #495057; line-height: 1.8; margin-bottom: 0;'>
                        <li><b>거시경제 지표와 거래량 분석:</b> 서울 지역 아파트 거래량에 미치는 주요 거시경제 요인 간 관계 규명</li>
                        <li><b>직관적 시각화 대시보드:</b> 변수 간 상관관계와 다중선형회귀 통계량을 가로 화면에서 한눈에 파악</li>
                        <li><b>실시간 시나리오 시뮬레이션:</b> 금리 및 가격지수 변화에 따른 실질 예상 거래량을 즉각 확인</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎯 프로젝트 수요층 (Target Users)")
            u1, u2, u3 = st.columns(3)
            with u1:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h5 style='color: #1f77b4; margin-bottom: 12px;'>1. 부동산 프롭테크 기획자·분석가</h5>
                    <ul style='color: #495057; line-height: 1.7; padding-left: 18px; margin-bottom: 0; font-size: 0.95em;'>
                        <li>금리 변화에 따른 시장 흐름을 직관적으로 보여주는 서비스 기획에 활용</li>
                        <li>사용자 대상 시뮬레이션 기능의 프로토타입으로 활용 가능</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with u2:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h5 style='color: #ff7f0e; margin-bottom: 12px;'>2. 부동산·금융 리서치 연구원</h5>
                    <ul style='color: #495057; line-height: 1.7; padding-left: 18px; margin-bottom: 0; font-size: 0.95em;'>
                        <li>금리 및 거시경제 변화가 거래량에 미치는 영향을 빠르게 분석</li>
                        <li>시장 민감도(Sensitivity) 분석을 위한 참고 자료로 활용</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with u3:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h5 style='color: #2ca02c; margin-bottom: 12px;'>3. 부동산 시장 참여자</h5>
                    <p style='color: #495057; line-height: 1.6; margin-bottom: 8px; font-size: 0.95em;'><b>공인중개사 · 개인 투자자 · 실수요자</b></p>
                    <p style='color: #6c757d; font-size: 0.9em; line-height: 1.6; margin-bottom: 0;'>
                        거래량은 시장 분위기를 보여주는 선행지표이므로, 금리와 가격 변화에 따른 시장 흐름을 이해하는 데 도움을 줍니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px 24px; border-radius: 12px; border-left: 5px solid #1f77b4; margin-bottom: 15px;'>
                <h5 style='color: #2C3E50; margin-bottom: 10px;'>💡 프로젝트 파이프라인 특징</h5>
                <p style='color: #495057; line-height: 1.8; margin-bottom: 0;'>
                    단순한 탐색적 데이터 분석(EDA)에서 끝나는 것이 아니라,<br>
                    <b>① 데이터 탐색(EDA) → ② 다중공선성(VIF) 검증 → ③ OLS 다중선형회귀분석 → ④ 회귀계수 해석 → ⑤ 시나리오 시뮬레이션</b>까지 
                    연결하여 분석 결과를 실제 사용자가 직접 활용할 수 있는 형태의 대시보드로 구현했습니다.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.info("🏠 **HouseLens**는 스크롤 없이 가로 와이드 화면 한눈에 경제지표와 아파트 매매 시장 흐름을 파악할 수 있도록 설계되었습니다. 좌측 사이드바 메뉴를 통해 세부 분석을 확인해 보세요!")

        with tab2:
            c_data, c_var = st.columns([1, 1.4])
            with c_data:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0;'>
                    <h4 style='color: #2C3E50; margin-bottom: 14px;'>📁 분석 데이터셋 개요</h4>
                    <p style='color: #495057; line-height: 1.8;'>
                        <b>• 대상 지역:</b> 서울특별시<br>
                        <b>• 분석 기간:</b> 2020년 01월 ~ 2026년 05월 (총 77개월)<br>
                        <b>• 포괄 시기 특성:</b><br>
                        코로나19 초저금리 시기, 글로벌 통화긴축 금리 인상기, 최근 시장 조정기를 모두 포함하여 다양한 거시경제 환경에서의 거래량 변동을 심층 분석합니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with c_var:
                var_df = pd.DataFrame([
                    {"변수명": "아파트 매매 거래량 (종속변수)", "출처": "국토교통부", "설명": "서울 월별 아파트 매매 건수 (분석 핵심 대상)"},
                    {"변수명": "기준금리 변화", "출처": "한국은행", "설명": "기준금리의 전월 대비 변화량 (.diff)"},
                    {"변수명": "주택담보대출금리 변화", "출처": "한국은행", "설명": "주택담보대출 금리의 전월 대비 변화량 (.diff)"},
                    {"변수명": "아파트 매매가격지수 변화율", "출처": "한국부동산원", "설명": "전월 대비 매매가격지수 변화율 (.pct_change)"},
                    {"변수명": "소비자물가지수 변화율", "출처": "통계청", "설명": "전월 대비 소비자물가지수 변화율 (.pct_change)"}
                ])
                st.markdown("#### 📌 사용 데이터 및 핵심 변수")
                st.dataframe(var_df, use_container_width=True, hide_index=True)

        with tab3:
            c_proc, c_sim = st.columns(2)
            with c_proc:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h4 style='color: #2C3E50; margin-bottom: 12px;'>🛠️ 피처 엔지니어링 & 회귀분석</h4>
                    <ul style='color: #495057; line-height: 1.8; margin-bottom: 0;'>
                        <li><b>금리 변수 정규화:</b> 원시 수준 대신 전월 대비 <b>변화량(.diff())</b>을 사용하여 실질 금리 변동 충격을 포착</li>
                        <li><b>지수 변수 정규화:</b> 지수 원본 대신 전월 대비 <b>변화율(.pct_change())</b>을 사용하여 상대적 등락을 반영</li>
                        <li><b>다중공선성 & 회귀모형:</b> VIF 검정을 통해 다중공선성을 배제하고 OLS 다중선형회귀분석 수행</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with c_sim:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h4 style='color: #2C3E50; margin-bottom: 12px;'>🔮 시나리오 시뮬레이션 활용법</h4>
                    <p style='color: #495057; line-height: 1.7;'>
                        <b>'시나리오 분석'</b> 메뉴에서는 <b>현재 시점(2026년 5월) 거래량</b>을 초기값(기준점)으로 설정합니다.<br><br>
                        슬라이더를 통해 주택담보대출금리 변화 및 아파트 매매가격지수 변화율을 가정하면, 모형의 회귀계수를 기반으로 
                        <b>예상 거래량과 실질 증감량</b>을 실시간 시뮬레이션할 수 있습니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            st.markdown("""
            <div style='margin-bottom: 15px;'>
                <p style='color: #495057; line-height: 1.7; font-size: 1.05em;'>
                    본 대시보드는 거시경제 핵심 지표와 아파트 매매 거래량 간의 정량적 관계를 규명하고 시뮬레이션하는 데 중점을 두었습니다.<br>
                    보다 종합적이고 정확한 시장 해석을 위해 다음의 <b>분석 모형 한계점(Limitations)</b>을 참고해 주시기 바랍니다.
                </p>
            </div>
            """, unsafe_allow_html=True)

            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 22px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%; margin-bottom: 15px;'>
                    <h5 style='color: #d62728; margin-bottom: 12px;'>1. 정책 및 외부 이벤트 미반영</h5>
                    <p style='color: #495057; line-height: 1.7; font-size: 0.95em; margin-bottom: 0;'>
                        <b>코로나19, 정부의 부동산 정책, 대출 규제(DSR), 세제 개편</b> 등과 같은 비계량적 요인은 분석 변수에 포함되지 않았습니다.<br><br>
                        시장 거시 이벤트 충격이 큰 시기에는 정량적 지표 외 비계량 요인이 실거래량에 큰 영향을 미칠 수 있습니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with row1_col2:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 22px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%; margin-bottom: 15px;'>
                    <h5 style='color: #d62728; margin-bottom: 12px;'>2. 서울 전체 합계 데이터 사용</h5>
                    <p style='color: #495057; line-height: 1.7; font-size: 0.95em; margin-bottom: 0;'>
                        본 분석은 <b>서울시 전체 거래량</b>을 대상으로 수행하였으며, <b>강남권·비강남권</b> 등 지역별 특성은 반영하지 않았습니다.<br><br>
                        자치구 및 상급지/비상급지 입지 여부에 따라 금리와 가격 변동에 대한 민감도가 다를 수 있습니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 22px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h5 style='color: #d62728; margin-bottom: 12px;'>3. 장기적인 시차 효과 미반영</h5>
                    <p style='color: #495057; line-height: 1.7; font-size: 0.95em; margin-bottom: 0;'>
                        금리 변화가 부동산 거래량에 미치는 영향은 즉시 나타나기보다 <b>수개월에 걸쳐 점진적으로 반영</b>될 수 있습니다.<br><br>
                        본 분석에서는 변수의 변화량과 변화율은 반영했지만, <b>1~3개월 이상의 장기 시차(Lag) 효과</b>는 별도로 고려하지 않았습니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with row2_col2:
                st.markdown("""
                <div style='background-color: #ffffff; padding: 22px; border-radius: 12px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h5 style='color: #d62728; margin-bottom: 12px;'>4. 단순 선형 관계 가정</h5>
                    <p style='color: #495057; line-height: 1.7; font-size: 0.95em; margin-bottom: 0;'>
                        <b>OLS 회귀모형</b>은 변수 간 선형 관계를 가정합니다.<br><br>
                        실제 부동산 시장은 다양한 비선형 요인과 복합적인 상호작용에 의해 움직일 수 있으므로, 모든 시장 상황을 완벽하게 설명하지는 않습니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **안내**: 본 시뮬레이션 결과는 미래 거래량에 대한 단정적 예측값이 아니라, 거시경제 변수 변화에 따른 거래량의 방향성과 민감도를 확인하기 위한 참고 자료로 활용하시기 바랍니다.")

    except Exception as e:
        st.error(f"Home 화면 오류: {e}")

def draw_eda(df):
    st.subheader("거래량 분석")
    if df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    col_charts, col_table = st.columns([1.3, 1.0])

    with col_charts:
        chart_tabs = st.tabs(["📈 월별 거래량 추이", "🏢 평균 매매가격지수 추이"])
        with chart_tabs[0]:
            fig1 = px.line(df, x='날짜', y='거래량', title="서울 아파트 월별 거래량 추이", markers=True)
            fig1.update_traces(line_color='#1f77b4', marker=dict(size=6))
            st.plotly_chart(fig1, use_container_width=True)
        with chart_tabs[1]:
            fig2 = px.line(df, x='날짜', y='아파트 매매가격지수', title="서울 아파트 평균 매매가격지수 추이", markers=True)
            fig2.update_traces(line_color='#ff7f0e', marker=dict(size=6))
            st.plotly_chart(fig2, use_container_width=True)

    with col_table:
        st.markdown("#### 📋 선택 기간 월별 수치 표")
        total_vol = int(df['거래량'].sum())
        avg_vol = int(df['거래량'].mean())
        kpi_sub1, kpi_sub2 = st.columns(2)
        kpi_sub1.metric("기간 총 거래량", f"{total_vol:,} 건")
        kpi_sub2.metric("월평균 거래량", f"{avg_vol:,} 건")

        df_sorted = df.sort_values('날짜').copy()
        summary_df = pd.DataFrame({
            '연월': df_sorted['날짜'].dt.strftime('%Y-%m'),
            '거래량(건)': df_sorted['거래량'],
            '매매가격지수': df_sorted['아파트 매매가격지수'],
            '기준금리(%)': df_sorted['기준금리'],
            '주담대금리(%)': df_sorted['주택담보대출금리']
        }).reset_index(drop=True)

        st.dataframe(summary_df.set_index('연월'), use_container_width=True, height=350)

def draw_economy(df):
    st.subheader("경제지표 분석")
    if df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    vars_to_corr = ['거래량', '기준금리', '주택담보대출금리', '아파트 매매가격지수', '소비자물가지수']
    existing_vars = [v for v in vars_to_corr if v in df.columns]

    col_heat1, col_heat2 = st.columns([2, 1])
    with col_heat1:
        corr_df = df[existing_vars].corr()
        fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="경제지표 상관관계 Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    with col_heat2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.info("**[히트맵 설명]**\n\n색의 농도는 변수 간 상관관계의 강도를 의미하며, 붉은색은 양(+)의 상관관계, 푸른색은 음(-)의 상관관계를 나타냅니다. 색이 진할수록 두 변수 간의 연관성이 더욱 강함을 의미합니다. 분석 결과, 거래량은 주택담보대출금리가 상승할수록 감소하는 경향을 보였으며, 기준금리, 주택담보대출금리, 소비자물가지수는 높은 양의 상관관계를 보여 거시경제 지표들이 서로 밀접하게 연계되어 움직이는 특성을 확인할 수 있었습니다.")

    st.markdown("### 거시경제 지표 vs 거래량")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**① 금리 그룹**")
        base_rate_selected = st.checkbox("기준금리", key="base_rate")
        mortgage_rate_selected = st.checkbox("주택담보대출금리", key="mortgage_rate")
    with col2:
        st.markdown("**② 지수 그룹**")
        price_index_selected = st.checkbox("아파트 매매가격지수", key="price_index")
        cpi_selected = st.checkbox("소비자물가지수", key="cpi")

    rate_group_selected = base_rate_selected or mortgage_rate_selected
    index_group_selected = price_index_selected or cpi_selected

    if rate_group_selected and index_group_selected:
        st.error("금리 그룹과 지수 그룹은 서로 섞어서 선택할 수 없습니다. 한 그룹 내에서만 선택해주세요.")
    else:
        selected_vars = []
        if base_rate_selected: selected_vars.append('기준금리')
        if mortgage_rate_selected: selected_vars.append('주택담보대출금리')
        if price_index_selected: selected_vars.append('아파트 매매가격지수')
        if cpi_selected: selected_vars.append('소비자물가지수')

        trend_agg = {'거래량': 'sum'}
        for v in selected_vars:
            trend_agg[v] = 'mean'

        trend_df = df.groupby('날짜').agg(trend_agg).reset_index()

        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Scatter(x=trend_df['날짜'], y=trend_df['거래량'], name="거래량", mode='lines+markers', line=dict(color='#1f77b4')), secondary_y=False)

        colors = ['#ff7f0e', '#2ca02c']
        for i, v in enumerate(selected_vars):
            fig_dual.add_trace(go.Scatter(x=trend_df['날짜'], y=trend_df[v], name=v, mode='lines+markers', line=dict(color=colors[i%len(colors)])), secondary_y=True)

        fig_dual.update_layout(title_text="거시경제 지표 vs 거래량 (선택 변수 비교)")
        fig_dual.update_yaxes(title_text="거래량", secondary_y=False)
        if selected_vars:
            fig_dual.update_yaxes(title_text=", ".join(selected_vars), secondary_y=True)

        st.plotly_chart(fig_dual, use_container_width=True)
def draw_regression(df):
    st.subheader("회귀분석 결과")

    Y = df['거래량']
    X = df.drop(columns=['거래량'])
    X_with_const = sm.add_constant(X)

    try:
        st.markdown("### 다중공선성(VIF) 검증")
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        vif_df = vif_data[vif_data["Feature"] != "const"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig_vif = px.bar(vif_df, x='Feature', y='VIF', title="독립변수별 VIF 지수", text_auto='.2f')
            fig_vif.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF=10 (경고선)")
            st.plotly_chart(fig_vif, use_container_width=True)

        st.info("VIF(분산팽창계수)는 변수들 간의 중복 정도(다중공선성)를 확인하는 지표입니다. 일반적으로 VIF가 5 이상이면 다중공선성을 의심하고, 10 이상이면 심각한 수준으로 판단합니다. 다중공선성이 높으면 어떤 변수가 결과에 얼마나 영향을 미치는지 정확하게 판단하기 어려워지고, 회귀모형의 신뢰성이 낮아질 수 있습니다. 분석 결과 모든 변수의 VIF가 약 1.1~1.3 수준으로 나타나 변수 간 중복이 거의 없었으며, 회귀분석을 수행하기에 적합한 것으로 확인되었습니다.")
        st.markdown("---")

        model = sm.OLS(Y, X_with_const).fit()

        st.markdown("### OLS 다중선형회귀분석")

        r2 = model.rsquared
        adj_r2 = model.rsquared_adj
        f_stat = model.fvalue
        p_f_stat = model.f_pvalue

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("R²", f"{r2:.4f}")
        kpi1.caption("거래량 변동의 약 43.1%를 설명합니다.")
        kpi2.metric("Adjusted R²", f"{adj_r2:.4f}")
        kpi2.caption("변수 개수를 고려한 실제 설명력입니다.")
        kpi3.metric("F-statistic", f"{f_stat:.4f}")
        kpi3.caption("모형 전체의 통계적 유의성을 평가합니다.")
        kpi4.metric("Prob(F-statistic)", f"{p_f_stat:.4e}")
        kpi4.caption("p<0.001로 모형 전체가 통계적으로 유의합니다.")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("OLS Summary 전체 보기"):
            st.text(model.summary())

        st.markdown("#### 회귀계수 및 유의성 검정")
        coef_df = pd.DataFrame({
            'Feature': model.params.index,
            'Coef': model.params.values,
            'P-value': model.pvalues.values
        })
        coef_df = coef_df[coef_df['Feature'] != 'const'].reset_index(drop=True)

        def highlight_pval(val):
            return 'background-color: #fffae6; color: red; font-weight: bold;' if val < 0.05 else ''

        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            try:
                styled_coef = coef_df.style.map(highlight_pval, subset=['P-value']).format({"Coef": "{:.4f}", "P-value": "{:.4e}"})
            except AttributeError:
                styled_coef = coef_df.style.applymap(highlight_pval, subset=['P-value']).format({"Coef": "{:.4f}", "P-value": "{:.4e}"})
            st.dataframe(styled_coef, use_container_width=True)

        with col_c2:
            coef_df['Color'] = np.where(coef_df['Coef'] > 0, '양수', '음수')
            fig_coef = px.bar(coef_df, x='Coef', y='Feature', orientation='h', 
                              color='Color', color_discrete_map={'양수': '#1f77b4', '음수': '#d62728'},
                              title="회귀계수 (양수: 파란색, 음수: 빨간색)")
            st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 변수별 상세 해석 및 회귀선")
        valid_features = [f for f in coef_df['Feature'].tolist() if f in df.columns]
        selected_var = st.radio("상세 해석을 확인할 변수를 클릭하세요:", valid_features, horizontal=True)

        if selected_var:
            var_row = coef_df[coef_df['Feature'] == selected_var].iloc[0]
            coef_val = var_row['Coef']
            pval = var_row['P-value']
            is_sig = pval < 0.05

            sig_text = "<span style='color: white; background-color: #28a745; padding: 2px 8px; border-radius: 10px; font-size: 0.9em;'>유의</span>" if is_sig else "<span style='color: white; background-color: #dc3545; padding: 2px 8px; border-radius: 10px; font-size: 0.9em;'>유의하지 않음</span>"

            desc = ""
            if "월" in selected_var:
                desc = "거래량과 유의한 관계가 확인되지 않았습니다. (p=0.528)"
            elif "기준금리" in selected_var:
                desc = "거래량 감소 방향의 계수가 나타났지만 통계적으로 유의하지 않았습니다. (Coef=-2722.72, p=0.218)"
            elif "주택담보대출" in selected_var:
                desc = "금리가 상승할수록 거래량이 감소하는 경향이 나타났으며, 통계적으로 유의했습니다. (Coef=-7893.04, p=0.001)"
            elif "소비자물가" in selected_var:
                desc = "거래량과 유의한 관계가 확인되지 않았습니다. (p=0.375)"
            elif "매매가격" in selected_var:
                desc = "지수 상승률이 높을수록 거래량이 증가하는 경향이 나타났으며, 통계적으로 유의했습니다. (Coef=221258.56, p<0.001)"
            else:
                desc = "해당 변수에 대한 설명이 없습니다."

            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                st.markdown(f"""
                <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; height: 100%;'>
                    <h5 style='margin-bottom: 15px;'>{selected_var} 상세 해석</h5>
                    <p style='margin-bottom: 5px;'><b>회귀계수 (Coef):</b> {coef_val:,.4f}</p>
                    <p style='margin-bottom: 5px;'><b>P-value:</b> {pval:.4e}</p>
                    <p style='margin-bottom: 15px;'><b>유의 여부:</b> {sig_text}</p>
                    <hr>
                    <p style='margin-top: 15px;'><b>해석:</b><br>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

            if selected_var in df.columns:
                with col_d2:
                    fig_scatter = px.scatter(df, x=selected_var, y='거래량', trendline='ols',
                                             title=f"{selected_var} vs 거래량 산점도 및 회귀선",
                                             labels={selected_var: selected_var, '거래량': '거래량'})
                    st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")

def draw_scenario(df):
    st.markdown("### 🔮 시나리오 분석 (What-If Simulation)")
    st.markdown("""
<div style='background-color: #f8f9fa; padding: 20px 24px; border-radius: 12px; border-left: 5px solid #1f77b4; margin-bottom: 20px;'>
    <h5 style='color: #2C3E50; margin-bottom: 12px;'>📌 시나리오 분석 안내</h5>
    <p style='color: #495057; line-height: 1.7; margin-bottom: 10px;'>
        본 시나리오 분석은 <b>미래를 정확히 예측하기 위한 목적이 아니라</b>, 주요 거시경제 지표 변화가 거래량에 미치는 영향을 탐색하기 위한 <b>What-if 시뮬레이션</b>입니다.<br>
        실제 부동산 거래량은 <b>정부 정책, 대출 규제, 시장 심리, 경기 상황, 공급 및 수요</b> 등 다양한 요인의 영향을 함께 받습니다.<br>
        따라서 본 시뮬레이션은 거시경제 변수 변화에 따른 <b>거래량의 방향성과 민감도를 확인하기 위한 참고 자료</b>로 활용하시기 바랍니다.
    </p>
    <p style='color: #6c757d; font-size: 0.9em; margin-bottom: 0;'>
        ※ 슬라이더를 통해 변화량을 설정하면 <b>현재(2026년 5월 기준)</b> 실거래량 대비 회귀모형 계수를 반영한 예상 거래량 및 증감률이 산출됩니다. (초기값 0.00 시 현재 실거래량 산출)
    </p>
</div>
""", unsafe_allow_html=True)

    Y = df['거래량']
    X = df.drop(columns=['거래량'])
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    coef = model.params

    def reset_scenario_sliders():
        st.session_state["scenario_mortgage"] = 0.0
        st.session_state["scenario_price_index"] = 0.0

    col_header, col_btn = st.columns([6, 1])
    with col_header:
        st.markdown("#### 경제지표 시나리오 설정")
    with col_btn:
        st.button("🔄 초기값", on_click=reset_scenario_sliders, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        mortgage_change = st.slider(
            "주담대금리 변화 (%p)",
            min_value=-0.50,
            max_value=0.50,
            value=0.00,
            step=0.01,
            format="%.2f",
            key="scenario_mortgage"
        )
    with col2:
        price_change = st.slider(
            "아파트매매가격지수 변화율 (%)",
            min_value=-0.03,
            max_value=0.03,
            value=0.00,
            step=0.001,
            format="%.3f",
            key="scenario_price_index"
        )

    current_vol = Y.iloc[-1]
    pred_vol = (
        current_vol
        + (mortgage_change * coef['주담대금리_변화'])
        + (price_change * coef['아파트매매가격지수_변화율'])
    )

    diff_vol = pred_vol - current_vol
    diff_pct = (diff_vol / current_vol) * 100 if current_vol != 0 else 0

    st.markdown("### 시뮬레이션 결과")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("예상 거래량", f"{int(pred_vol):,} 건")
    kpi2.metric("현재 대비 증감량", f"{int(diff_vol):,} 건", delta=f"{int(diff_vol)}", delta_color="normal")
    kpi3.metric("증감률", f"{diff_pct:.2f} %", delta=f"{diff_pct:.2f}%", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        bar_data = pd.DataFrame({
            '구분': ['현재 거래량', '예상 거래량'],
            '거래량': [current_vol, pred_vol]
        })
        fig_bar = px.bar(bar_data, x='구분', y='거래량', color='구분', title="현재 vs 예상 거래량", text_auto='.2s', 
                         color_discrete_map={'현재 거래량':'#7f7f7f', '예상 거래량':'#1f77b4'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        max_gauge = max(current_vol, pred_vol) * 1.5 if max(current_vol, pred_vol) > 0 else 10000
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred_vol,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "예상 거래량 수준"},
            delta = {'reference': current_vol, 'position': "top"},
            gauge = {
                'axis': {'range': [0, max_gauge]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, current_vol], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_vol
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("### 🤖 AI 시장 리포트")
    report_lines = []

    if diff_vol > 0:
        report_lines.append(f"**💡 전체 전망**: 설정하신 조건에 따르면 거래량은 현재 대비 **{diff_pct:.1f}% 증가**할 것으로 예상됩니다.")
    elif diff_vol < 0:
        report_lines.append(f"**💡 전체 전망**: 설정하신 조건에 따르면 거래량은 현재 대비 **{abs(diff_pct):.1f}% 감소**할 것으로 예상됩니다.")
    else:
        report_lines.append(f"**💡 전체 전망**: 설정하신 조건에서는 거래량이 현재 수준을 **유지**할 것으로 예상됩니다.")

    if mortgage_change != 0:
        direction = "상승" if mortgage_change > 0 else "하락"
        effect = "감소" if (mortgage_change * coef['주담대금리_변화']) < 0 else "증가"
        report_lines.append(f"- **주담대금리 변화 ({mortgage_change:+.2f}%p {direction})**: 주택담보대출금리 변화로 인해 아파트 거래량이 {effect}하는 요인으로 작용하고 있습니다.")

    if price_change != 0:
        direction = "상승" if price_change > 0 else "하락"
        effect = "증가" if (price_change * coef['아파트매매가격지수_변화율']) > 0 else "감소"
        report_lines.append(f"- **아파트매매가격지수 변화율 ({price_change:+.3f}% {direction})**: 아파트 매매가격지수 변화율로 인해 아파트 거래량이 {effect}하는 요인으로 작용하고 있습니다.")

    if len(report_lines) == 1:
        report_lines.append(f"- 현재 설정된 변화값이 모두 0.00이므로, 현재(2026년 5월) 기준 실거래량({int(current_vol):,}건)이 예상 거래량으로 산출되었습니다.")

    st.info("\n\n".join(report_lines))

def main():
    apply_custom_css()
    st.title("🏠 HouseLens: 거시경제 지표 기반 서울 아파트 거래량 분석 및 시나리오 대시보드")
    st.markdown("---")

    merged_df = load_merged_data()
    model_df = load_model_data()

    filtered_df, menu = configure_sidebar(merged_df)

    if menu == "Home":
        draw_home(filtered_df)
    elif menu == "거래량 분석":
        draw_eda(filtered_df)
    elif menu == "경제지표 분석":
        draw_economy(filtered_df)
    elif menu == "회귀분석 결과":
        draw_regression(model_df)
    elif menu == "시나리오 분석":
        draw_scenario(model_df)

if __name__ == "__main__":
    main()
