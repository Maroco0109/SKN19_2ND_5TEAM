def create_comparison_chart(pred_time, time_val):
  import plotly.graph_objects as go

  # --- 1. 사용할 두 점의 값을 직접 지정 ---
  pred_point = pred_time
  real_point = time_val

  # --- 2. Plotly를 사용한 시각화 ---
  fig = go.Figure()

  # 축 역할을 하는 얇은 회색 선 추가

  fig.add_shape(
      type='line',
      x0=0, y0=0, x1=0, y1=95,
      line=dict(color='lightgrey', width=1),
      layer='below'  # 이 라인을 추가하여 선을 맨 뒤로 보냅니다.
  )
  # 두 점(마커) 추가
  fig.add_trace(go.Scatter(
      x=[0],
      y=[pred_point],
      mode='text',
      text=[f'<b>{pred_point}</b>▶'],  # 텍스트를 굵게 표시
      textposition='middle left',
      name='예측 생존 기간',  # 범례에 표시될 이름
      marker=dict(color="#7fbdff"),  # 범례의 색상 아이콘을 위해 필요
      textfont=dict(
          size=22,
          color="#7fbdff"  # 텍스트 색상을 범례 색상과 맞춤
      ),
      hoverinfo='none'
  ))

  # 2. 실제값 Trace 추가
  fig.add_trace(go.Scatter(
      x=[0],
      y=[real_point],
      mode='text',
      text=[f'◀<b>{real_point}</b>'], # 텍스트를 굵게 표시
      textposition='middle right',
      name='실제 생존 기간',  # 범례에 표시될 이름
      marker=dict(color="#ff7e7e"), # 범례의 색상 아이콘을 위해 필요
      textfont=dict(
          size=22,
          color="#ff7e7e"  # 텍스트 색상을 범례 색상과 맞춤
      ),
      hoverinfo='none'
  ))

  # --- 3. 차트 레이아웃 꾸미기 ---
  fig.update_layout(
      title=dict(text="<b>실제 생존 기간과 예측 생존 기간 비교</b>", x=0.5, font=dict(size=20)),
      # X축 설정
      xaxis=dict(
          visible= False,
      ),
      # Y축 설정 수정
      yaxis=dict(
          title="생존 기간(단위 : 3개월)",
          range=[-3, max(pred_point, real_point) + 15],
          showgrid=True,
          zeroline=True,
          griddash='dot',  # 이 라인을 추가하여 격자를 점선으로 변경
          gridcolor='lightgrey' # 격자 색상을 지정하면 더 잘 보입니다 (선택 사항)
      ),
      showlegend=True, 
      height=550,
      plot_bgcolor='white',
      margin=dict(t=80),
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1
      )
  )

  # --- Streamlit에 차트 표시 ---
  st.plotly_chart(fig, use_container_width=True)