import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'

# Set Streamlit page configuration
st.set_page_config(
	page_title="النماذج القياسية الأكثر شهرة عربيا",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Apply basic RTL styling
st.markdown("""
<style>
    /* Basic RTL direction for layout */
    * {
        direction: rtl;
    }
    .main .block-container, h1, h2, h3, h4, h5, h6, p, div {
        direction: rtl;
        text-align: right;
    }
    .katex-display {
        direction: ltr !important;
    }
</style>
""", unsafe_allow_html=True)


# Main title
st.markdown("# النماذج القياسية الأكثر شهرة عربياً")
st.markdown("### إعداد: Merwan Roudane")

# Sidebar
st.sidebar.markdown("### قائمة النماذج")
options = [
	"الرئيسية",
	"نموذج الانحدار الخطي وفروعه",
	"نموذج الانحدار الكمي",
	"نموذج المعادلات الآنية",
	"نموذج VAR",
	"نموذج VECM",
	"نموذج ARDL",
	"نموذج NARDL",
	"نماذج البانل الديناميكية",
	"نماذج البانل الساكنة",
	"المتناقضات في الدراسات العربية",
	"ملاحظات عامة"
]
choice = st.sidebar.radio("اختر النموذج:", options)

# Additional info in the sidebar
st.sidebar.markdown("---")
st.sidebar.info("هذا المخطط يتناول أهم النماذج التفسيرية في الدراسات العربية وليست التنبؤية")
st.sidebar.info("الشروط المذكورة هي بصفة عامة وليست مفصلة حيث تحتاج الشروط المفصلة إلى مخطط لكل نموذج على حدى")


# Function to create a model tree graph
def create_model_tree():
	fig = go.Figure()

	models = [
		"نماذج الانحدار الخطي", "نماذج الانحدار الكمي", "نماذج المعادلات الآنية",
		"نموذج VAR", "نموذج VECM", "نموذج ARDL", "نموذج NARDL",
		"نماذج البانل الديناميكية", "نماذج البانل الساكنة"
	]

	x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	y = [3, 2, 3, 1, 1, 2, 2, 3, 3]

	# Add nodes
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode='markers+text',
		marker=dict(size=20, color=['#3a506b'] * len(models)),
		text=models,
		textposition="top center",
		textfont=dict(size=14, color='black', family='Arial'),
		hoverinfo='text'
	))

	# Add connecting lines
	fig.add_shape(type="line", x0=1, y0=3, x1=3, y1=3, line=dict(color="#718096", width=2))
	fig.add_shape(type="line", x0=4, y0=1, x1=7, y1=1, line=dict(color="#718096", width=2))
	fig.add_shape(type="line", x0=8, y0=3, x1=9, y1=3, line=dict(color="#718096", width=2))

	# Format graph
	fig.update_layout(
		title={
			'text': "ترابط النماذج القياسية",
			'y': 0.95,
			'x': 0.5,
			'xanchor': 'center',
			'yanchor': 'top',
			'font': dict(size=24)
		},
		xaxis=dict(
			showticklabels=False,
			showgrid=False,
			zeroline=False,
		),
		yaxis=dict(
			showticklabels=False,
			showgrid=False,
			zeroline=False,
		),
		showlegend=False,
		height=500,
		plot_bgcolor='#f9f9f9',
	)

	return fig


# Function to create a conditions comparison graph
def create_conditions_comparison():
	categories = ['استقرارية البيانات', 'حجم العينة', 'التوزيع الطبيعي', 'مشاكل التوصيف', 'العلاقة السببية']

	models = ['ARDL', 'VAR', 'VECM', 'نماذج البانل']
	values = [
		[3, 3, 2, 4, 5],  # ARDL
		[5, 4, 3, 3, 5],  # VAR
		[5, 4, 2, 3, 5],  # VECM
		[4, 5, 2, 4, 3],  # Panel Models
	]

	fig = go.Figure()

	for i, model in enumerate(models):
		fig.add_trace(go.Scatterpolar(
			r=values[i],
			theta=categories,
			fill='toself',
			name=model
		))

	fig.update_layout(
		polar=dict(
			radialaxis=dict(
				visible=True,
				range=[0, 5]
			)
		),
		showlegend=True,
		title={
			'text': "مقارنة شروط النماذج القياسية",
			'y': 0.95,
			'x': 0.5,
			'xanchor': 'center',
			'yanchor': 'top',
			'font': dict(size=24)
		},
		height=500
	)

	return fig


# Home Page
if choice == "الرئيسية":
	st.markdown("## مقدمة عن النماذج القياسية الشائعة الاستخدام عربياً")

	st.info(
		"تقدم هذه الوثيقة عرضاً للنماذج القياسية الأكثر شيوعاً في الدراسات العربية مع توضيح الشروط الأساسية لاستخدامها. تشمل هذه النماذج أنواعاً مختلفة من تحليل الانحدار، ونماذج المعادلات الآنية، ونماذج السلاسل الزمنية، ونماذج البانل."
	)

	st.plotly_chart(create_model_tree(), use_container_width=True)
	st.plotly_chart(create_conditions_comparison(), use_container_width=True)

	st.markdown("""
    ### أهمية اختيار النموذج المناسب
    يعتمد اختيار النموذج المناسب على عدة عوامل أهمها:
    - هدف الدراسة (تفسيري أم تنبؤي)
    - طبيعة البيانات (مقطعية، سلاسل زمنية، بيانات بانل)
    - خصائص المتغيرات (استقرارية، توزيع، إلخ)
    - العلاقة بين المتغيرات (أحادية الاتجاه، تبادلية)
    """)

# Linear Regression Model
elif choice == "نموذج الانحدار الخطي وفروعه":
	st.header("نموذج الانحدار الخطي وفروعه")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة الأثر المباشر للمتغيرات المستقلة على المتغير التابع.

        ### الشروط الأساسية
        - المتغير التابع يكون continuous ويتبع التوزيع الطبيعي
        - في النمذجة التقليدية، يكون حجم العينة أكبر من عدد المتغيرات المستقلة بكثير
        - في النمذجة الحديثة، لا يشترط هذا الشرط
        - غياب مشاكل التوصيف
        - طريقة التقدير OLS تتطلب التحقق من الفرضيات الكلاسيكية

        ### الصيغة الرياضية
        """)

		st.latex(r"Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + ... + \beta_k X_{ki} + \varepsilon_i")

		st.markdown("""
        ### البدائل في حالات خاصة
        - في وجود التواء من جهة اليمين للمتغير التابع: استخدام Gamma regression أو Quantile regression
        - في وجود التواء من جهة اليسار للمتغير التابع: استخدام Skewed regression أو Quantile regression
        - في حالة وجود نقاط شاذة: استخدام Robust regression
        - في حالة المتغير التابع عبارة عن count variable: استخدام نماذج مثل Poisson، Binomial
        - في حالة المتغير التابع عبارة عن متغير ثنائي: استخدام نماذج مثل Logistic، Probit
        - في حالة المتغير التابع عبارة عن فئات: استخدام Categorical regression
        - في حالة المتغير التابع عبارة عن مجال محدد: استخدام Interval-valued regression
        """)

	with col2:
		# Visualization
		fig = go.Figure()
		np.random.seed(42)
		x = np.linspace(0, 10, 100)
		y = 2 * x + 1 + np.random.normal(0, 2, 100)
		fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='البيانات', marker=dict(color='#3a506b', size=8)))
		coef = np.polyfit(x, y, 1)
		line = coef[0] * x + coef[1]
		fig.add_trace(go.Scatter(x=x, y=line, mode='lines', name='خط الانحدار', line=dict(color='#f05454', width=3)))
		fig.update_layout(title="مثال على الانحدار الخطي البسيط", xaxis_title="المتغير المستقل", yaxis_title="المتغير التابع", legend_title="البيانات", height=400)
		st.plotly_chart(fig, use_container_width=True)

		# Code Example
		st.markdown("### مثال على بنية نموذج الانحدار المتعدد")
		code = """
        import statsmodels.api as sm
        import pandas as pd

        # Load data
        df = pd.read_csv('data.csv')

        # Define variables
        X = df[['x1', 'x2', 'x3']]
        X = sm.add_constant(X)
        y = df['y']

        # Fit model
        model = sm.OLS(y, X).fit()

        # Print summary
        print(model.summary())
        """
		st.code(code, language='python')

# Quantile Regression Model
elif choice == "نموذج الانحدار الكمي":
	st.header("نموذج الانحدار الكمي (Quantile Regression)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        - تقدير أثر المتغير المستقل على مختلف quantiles للمتغير التابع
        - البحث عن الأثر غير المتماثل لتأثير المتغير المستقل على المتغير التابع عند مختلف رتب quantile

        ### الشروط والخصائص
        - يستخدم في حالة وجود نقاط شاذة والتواء في المتغير التابع وحتى في المتغيرات المستقلة
        - مناسب عند وجود اختلافات وفروقات بين قيم المتغير التابع داخل العينة (مثل متغير الأجور أو الثروة)
        - يستخدم عند عدم التوزيع الطبيعي للبواقي في الانحدار العادي
        - مناسب عند الرغبة في الحصول على تفسيرات لا تتعلق بالمتوسط

        ### الصيغة الرياضية
        """)
		st.latex(r"Q_{Y}(\tau|X) = \beta_0(\tau) + \beta_1(\tau) X_1 + \beta_2(\tau) X_2 + ... + \beta_k(\tau) X_k")
		st.markdown("حيث τ هي رتبة الكمية (quantile) التي نهتم بها، وتتراوح من 0 إلى 1.")
		st.markdown("""
        ### تفرعات هذا النموذج
        - **Quantile in Quantile Regression:** نموذج أكثر مرونة يسمح بدراسة العلاقة بين الكميات للمتغيرات المستقلة والتابعة

        ### ميزات استخدام الانحدار الكمي
        - أقل تأثراً بالقيم المتطرفة مقارنة بالانحدار العادي
        - يسمح بتحليل تأثير المتغيرات المستقلة على كامل توزيع المتغير التابع وليس فقط على متوسطه
        - لا يتطلب افتراضات قوية حول توزيع البواقي
        """)

	with col2:
		# Visualization
		np.random.seed(42)
		x = np.linspace(0, 10, 200)
		y = 2 * x + 1 + np.random.exponential(scale=2, size=200)
		q_25 = 2 * x + 0.2
		q_50 = 2 * x + 1
		q_75 = 2 * x + 2.5
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='البيانات', marker=dict(color='#3a506b', size=6, opacity=0.7)))
		fig.add_trace(go.Scatter(x=x, y=q_25, mode='lines', name='الكمية 0.25', line=dict(color='#f05454', width=2)))
		fig.add_trace(go.Scatter(x=x, y=q_50, mode='lines', name='الكمية 0.50 (الوسيط)', line=dict(color='#30475e', width=2)))
		fig.add_trace(go.Scatter(x=x, y=q_75, mode='lines', name='الكمية 0.75', line=dict(color='#7b68ee', width=2)))
		fig.update_layout(title="مثال على الانحدار الكمي", xaxis_title="المتغير المستقل", yaxis_title="المتغير التابع", legend_title="البيانات والكميات", height=400)
		st.plotly_chart(fig, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق الانحدار الكمي")
		code = """
        import statsmodels.formula.api as smf
        import pandas as pd

        # Load data
        df = pd.read_csv('data.csv')

        # Fit quantile regression models
        q_25 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.25)
        q_50 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.50)
        q_75 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.75)

        # Print summary for the median
        print(q_50.summary())
        """
		st.code(code, language='python')

# Simultaneous Equations Model
elif choice == "نموذج المعادلات الآنية":
	st.header("نموذج المعادلات الآنية (Simultaneous Equations)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة العلاقات المتشابكة بين المتغيرات وتأثيرها الآني. حيث يمكن للمتغير أن يكون مستقلاً في معادلة وتابعاً في معادلة أخرى.

        ### الشروط الأساسية
        - وجود خاصية Simultaneity أي المتغير المستقل في المعادلة الأولى يصبح متغير تابع في المعادلة الثانية
        - تحقق شرط Order and Rank Conditions for Identification

        ### ملاحظات مهمة
        - في حالة استخدام هذا النموذج على السلاسل الزمنية غير المستقرة وفق طرق التقدير المعروفة، فإن Estimators تفقد الكفاءة (efficiency)
        - في حالة متغيرات غير مستقرة ومتكاملة، نستخدم منهجية Hisao 1997

        ### الصيغة الرياضية لنظام المعادلات الآنية
        """)
		st.latex(r"""
        \begin{align}
        Y_1 &= \beta_{10} + \beta_{12}Y_2 + \gamma_{11}X_1 + \gamma_{12}X_2 + \varepsilon_1 \\
        Y_2 &= \beta_{20} + \beta_{21}Y_1 + \gamma_{21}X_1 + \gamma_{22}X_2 + \varepsilon_2
        \end{align}
        """)
		st.markdown("""
        ### طرق التقدير
        - Two-Stage Least Squares (2SLS)
        - Three-Stage Least Squares (3SLS)
        - Limited Information Maximum Likelihood (LIML)
        - Full Information Maximum Likelihood (FIML)
        - Generalized Method of Moments (GMM)

        ### مثال على نظام معادلات آنية
        نموذج العرض والطلب في الاقتصاد:
        """)
		st.latex(r"""
        \begin{align}
        Q^d &= \alpha_0 + \alpha_1 P + \alpha_2 Y + \varepsilon_1 \quad \text{(معادلة الطلب)} \\
        Q^s &= \beta_0 + \beta_1 P + \beta_2 W + \varepsilon_2 \quad \text{(معادلة العرض)} \\
        Q^d &= Q^s \quad \text{(شرط التوازن)}
        \end{align}
        """)
		st.info("""
        حيث:
        - Q^d: الكمية المطلوبة
        - Q^s: الكمية المعروضة
        - P: السعر (متغير داخلي)
        - Y: الدخل (متغير خارجي يؤثر على الطلب)
        - W: تكلفة الإنتاج (متغير خارجي يؤثر على العرض)
        """)

	with col2:
		# Visualization
		nodes = ['Y₁', 'Y₂', 'X₁', 'X₂']
		edges = [('Y₁', 'Y₂'), ('Y₂', 'Y₁'), ('X₁', 'Y₁'), ('X₁', 'Y₂'), ('X₂', 'Y₁'), ('X₂', 'Y₂')]
		G = {node: [] for node in nodes}
		for edge in edges: G[edge[0]].append(edge[1])
		pos = {'Y₁': [0, 0.5], 'Y₂': [1, 0.5], 'X₁': [0.25, 1], 'X₂': [0.75, 1]}
		fig = go.Figure()
		for source, targets in G.items():
			for target in targets:
				fig.add_trace(go.Scatter(x=[pos[source][0], pos[target][0]], y=[pos[source][1], pos[target][1]], mode='lines', line=dict(width=2, color='#718096'), hoverinfo='none'))
		node_x = [pos[node][0] for node in nodes]
		node_y = [pos[node][1] for node in nodes]
		colors = ['#f05454', '#f05454', '#30475e', '#30475e']
		fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', marker=dict(size=30, color=colors, line=dict(width=2, color='white')), text=nodes, textposition="middle center", textfont=dict(size=20, color='white'), hoverinfo='text', hovertext=["المتغير التابع في المعادلة الأولى", "المتغير التابع في المعادلة الثانية", "متغير مستقل خارجي", "متغير مستقل خارجي"]))
		fig.update_layout(title="العلاقات المتشابكة في نموذج المعادلات الآنية", showlegend=False, height=400, plot_bgcolor='#f9f9f9', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.1, 1.1]), yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.4, 1.1]))
		st.plotly_chart(fig, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق نموذج المعادلات الآنية")
		code = """
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        import pandas as pd

        # Load data
        df = pd.read_csv('data.csv')

        # Define variables
        endog = df['y1']
        exog = sm.add_constant(df['y2'])
        instruments = sm.add_constant(df[['x1', 'x2']])

        # Fit 2SLS model
        model = IV2SLS(endog, exog, instruments).fit()

        # Print summary
        print(model.summary())
        """
		st.code(code, language='python')

# VAR Model
elif choice == "نموذج VAR":
	st.header("نموذج VAR (Vector Autoregression)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة العلاقة التبادلية بين المتغيرات في إطار السلاسل الزمنية، حيث يكون الهدف الأساسي هو التنبؤ بالإضافة إلى تحليل الصدمات. في هذا النموذج، تعتبر كل المتغيرات تابعة.

        ### الشروط المتعلقة بالاستقرارية
        - **المنهجية التقليدية:**
            - كل المتغيرات مستقرة في الفرق الأول أو الفرق الثاني وعدم وجود تكامل مشترك (أو عدم صلاحية نموذج VECM)
            - كل المتغيرات مستقرة في المستوى في إطار نظام من المعادلات
        - **المنهجية الحديثة:**
            - تطورات للنموذج حيث لا يشترط أصلاً دراسة الاستقرارية في إطار VAR-Integrated أو VAR-TVP

        ### أنواع وتعديلات النموذج
        - في وجود متغيرات مستقلة، ننتقل من VAR إلى VARx
        - إذا كان الهدف تحليل الصدمات، يمكن استخدام SVAR (Structural VAR)

        ### الصيغة الرياضية
        """)
		st.latex(r"""
        \mathbf{y}_t = \mathbf{c} + \mathbf{\Phi}_1 \mathbf{y}_{t-1} + \dots + \mathbf{\Phi}_p \mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t
        """)
		st.markdown("""
        ### استخدامات النموذج
        - التنبؤ بالقيم المستقبلية للمتغيرات
        - تحليل الصدمات وتأثيرها على المتغيرات
        - تحليل تفكيك التباين (Variance Decomposition)
        - تحليل دوال الاستجابة النبضية (Impulse Response Functions)
        """)

	with col2:
		# IRF Visualization
		fig_irf = go.Figure()
		periods = list(range(11))
		irf_values = [0, 0.05, 0.1, 0.14, 0.16, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01]
		confidence_upper = [v + 0.05 for v in irf_values]
		confidence_lower = [max(0, v - 0.05) for v in irf_values]
		fig_irf.add_trace(go.Scatter(x=periods + periods[::-1], y=confidence_upper + confidence_lower[::-1], fill='toself', fillcolor='rgba(58, 80, 107, 0.2)', line=dict(color='rgba(255, 255, 255, 0)'), hoverinfo='skip', showlegend=False))
		fig_irf.add_trace(go.Scatter(x=periods, y=irf_values, mode='lines+markers', line=dict(color='#3a506b', width=3), marker=dict(size=8), name='دالة الاستجابة النبضية'))
		fig_irf.add_shape(type='line', x0=0, y0=0, x1=10, y1=0, line=dict(color='#718096', width=1, dash='dash'))
		fig_irf.update_layout(title="مثال على دالة الاستجابة النبضية (IRF)", xaxis_title="الفترات الزمنية", yaxis_title="استجابة المتغير", height=300)
		st.plotly_chart(fig_irf, use_container_width=True)

		# FEVD Visualization
		fig_fevd = go.Figure()
		periods = list(range(1, 11))
		var1 = [100, 90, 80, 75, 70, 68, 65, 63, 60, 58]
		var2 = [0, 5, 10, 12, 15, 16, 18, 19, 21, 22]
		var3 = [0, 5, 10, 13, 15, 16, 17, 18, 19, 20]
		fig_fevd.add_trace(go.Bar(x=periods, y=var1, name='المتغير 1', marker_color='#3a506b'))
		fig_fevd.add_trace(go.Bar(x=periods, y=var2, name='المتغير 2', marker_color='#f05454'))
		fig_fevd.add_trace(go.Bar(x=periods, y=var3, name='المتغير 3', marker_color='#30475e'))
		fig_fevd.update_layout(title="مثال على تفكيك التباين", xaxis_title="الفترات الزمنية", yaxis_title="نسبة المساهمة (%)", barmode='stack', height=300)
		st.plotly_chart(fig_fevd, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق نموذج VAR")
		code = """
        import pandas as pd
        from statsmodels.tsa.api import VAR

        # Load data
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # Select optimal lag order
        model = VAR(df)
        results_order = model.select_order(maxlags=10)
        
        # Fit VAR model
        var_model = model.fit(results_order.aic)

        # Forecast
        forecast = var_model.forecast(df.values[-var_model.k_ar:], steps=5)

        # Impulse Response Analysis
        irf = var_model.irf(10)
        irf.plot()

        # Forecast Error Variance Decomposition
        fevd = var_model.fevd(10)
        fevd.plot()
        """
		st.code(code, language='python')

# VECM Model
elif choice == "نموذج VECM":
	st.header("نموذج VECM (Vector Error Correction Model)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة العلاقة التبادلية بين المتغيرات المعتمدة على الأجلين القصير والطويل في إطار نظام من المعادلات.

        ### الشروط الأساسية
        - يجب أن تكون كل المتغيرات مستقرة في الفرق الأول أو كلها في الفرق الثاني
        - يجب أن تتحقق شروط identification
        - يجب تحقق شروط متعلقة بـ exogeneity of variables
        - يجب أن يكون معامل تصحيح الخطأ سالب ومعنوي

        ### أنواع وتعديلات النموذج
        - في حالة وجود متغيرات مستقلة، يصبح نموذج VECM بـ VECMX
        - إذا كان هدف الدراسة هو تحليل الصدمات، يمكن الانتقال إلى SVECM

        ### الصيغة الرياضية
        """)
		st.latex(r"""
        \Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \varepsilon_t
        """)
		st.markdown("حيث:")
		st.latex(r"""
        \begin{align}
        \alpha &: \text{مصفوفة معاملات التعديل (معاملات تصحيح الخطأ)} \\
        \beta &: \text{مصفوفة متجهات التكامل المشترك} \\
        \Gamma_i &: \text{مصفوفة معاملات الآثار قصيرة الأجل}
        \end{align}
        """)
		st.markdown("""
        ### العلاقة بين VAR و VECM
        يمكن اعتبار VECM حالة خاصة من نموذج VAR مع قيود على المعاملات طويلة الأجل. وتحديداً، VECM هو نموذج VAR مقيد بوجود علاقة تكامل مشترك بين المتغيرات.

        ### مراحل تطبيق نموذج VECM
        1. اختبار استقرارية السلاسل الزمنية والتأكد من أنها متكاملة من الدرجة الأولى I(1)
        2. تحديد العدد الأمثل للفجوات الزمنية باستخدام معايير المعلومات
        3. اختبار وجود تكامل مشترك باستخدام منهجية جوهانسن
        4. تقدير نموذج VECM
        5. اختبار صلاحية النموذج من خلال فحص البواقي ومعامل تصحيح الخطأ
        """)

	with col2:
		# Visualization
		np.random.seed(42)
		t = np.linspace(0, 10, 200)
		equilibrium = 2 * t
		y1 = equilibrium + np.random.normal(0, 1, 200)
		y2 = equilibrium + np.random.normal(0, 1, 200)
		shock_point = 100
		y1[shock_point:shock_point + 30] += np.linspace(0, 5, 30)
		y1[shock_point + 30:] += 5 - 5 * np.exp(-0.1 * np.arange(70))
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=t, y=y1, mode='lines', name='السلسلة الزمنية 1', line=dict(color='#3a506b', width=2)))
		fig.add_trace(go.Scatter(x=t, y=y2, mode='lines', name='السلسلة الزمنية 2', line=dict(color='#f05454', width=2)))
		fig.add_trace(go.Scatter(x=t, y=equilibrium, mode='lines', name='التوازن طويل الأجل', line=dict(color='#30475e', width=2, dash='dash')))
		fig.add_annotation(x=t[shock_point], y=y1[shock_point], text="الصدمة", showarrow=True, arrowhead=1, ax=0, ay=-40)
		fig.add_annotation(x=t[shock_point + 50], y=y1[shock_point + 50], text="تصحيح الخطأ", showarrow=True, arrowhead=1, ax=0, ay=-40)
		fig.update_layout(title="آلية عمل نموذج تصحيح الخطأ (VECM)", xaxis_title="الزمن", yaxis_title="القيمة", height=400)
		st.plotly_chart(fig, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق نموذج VECM")
		code = """
        import pandas as pd
        from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

        # Load data
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # Johansen cointegration test
        johansen_test = coint_johansen(df, deterministic_order=0, k_ar_diff=2)

        # Determine number of cointegrating relationships
        coint_rank = johansen_test.rank

        # Fit VECM model
        model = VECM(df, k_ar_diff=2, coint_rank=coint_rank, deterministic='ci')
        results = model.fit()

        # Print summary
        print(results.summary())
        """
		st.code(code, language='python')

# ARDL Model
elif choice == "نموذج ARDL":
	st.header("نموذج ARDL (Autoregressive Distributed Lag)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة التأثير الديناميكي والعلاقة طويلة الأجل مع تقدير قصيرة الأجل في إطار معادلة واحدة (لا يوجد feedback بين المتغير التابع والمتغيرات المستقلة).

        ### الشروط الأساسية
        - الاستقرارية في المستوى أو الفرق الأول على الأكثر (لا توجد متغيرات مستقرة في الفرق الثاني)
        - حجم العينة على الأقل 30
        - في حالة حجم العينة أقل من 30، نستخدم ARDL BOOTSTRAPPING

        ### أنواع وتعديلات النموذج
        - في حالة المتغير التابع مستقر في المستوى، نستخدم AUGMENTED ARDL
        - في وجود عدة تغيرات هيكلية، نستخدم FOURRIER ARDL أو استخدام DUMMIES
        - في حالة عدم وجود علاقة طويلة الأجل، يمكن استخدام DIFFERENCED ARDL كبديل

        ### الصيغة الرياضية
        """)
		st.latex(r"""
        \Delta y_t = \alpha_0 + \delta y_{t-1} + \theta' \mathbf{x}_{t-1} + \sum_{i=1}^{p-1} \phi_i \Delta y_{t-i} + \sum_{j=0}^{q-1} \boldsymbol{\beta}_j' \Delta \mathbf{x}_{t-j} + \varepsilon_t
        """)
		st.markdown("""
        ### مزايا نموذج ARDL
        - يمكن استخدامه مع متغيرات ذات درجات تكامل مختلفة (I(0) و I(1) ولكن ليس I(2))
        - يسمح بتقدير العلاقات طويلة وقصيرة الأجل في معادلة واحدة
        - يعالج مشكلة Endogeneity وارتباط البواقي من خلال إدراج عدد كافٍ من الفجوات الزمنية
        - يمكن استخدامه مع عينات صغيرة نسبياً

        ### اختبارات الحدود (Bounds Test)
        يستخدم اختبار الحدود ARDL Bounds Test للتحقق من وجود علاقة توازن طويلة الأجل بين المتغيرات.
        - **الفرضية الصفرية:** لا توجد علاقة توازن طويلة الأجل.
        - **الفرضية البديلة:** توجد علاقة توازن طويلة الأجل.
        
        ### مراحل تطبيق نموذج ARDL
        1. التأكد من استقرارية المتغيرات (I(0) أو I(1) وليس I(2))
        2. تحديد العدد الأمثل للفجوات الزمنية
        3. تقدير نموذج ARDL و إجراء اختبار الحدود
        4. تقدير العلاقة طويلة الأجل ونموذج تصحيح الخطأ (إذا وجدت علاقة)
        5. إجراء اختبارات التشخيص للتحقق من صلاحية النموذج
        """)

	with col2:
		# Visualization
		fig = go.Figure()
		f_stat = 5.2
		lower_bound_1 = 2.8
		upper_bound_1 = 3.8
		lower_bound_5 = 2.1
		upper_bound_5 = 3.0
		fig.add_trace(go.Scatter(x=['القيمة المحسوبة'], y=[f_stat], mode='markers', marker=dict(size=15, color='#f05454'), name='إحصائية F المحسوبة'))
		fig.add_trace(go.Scatter(x=['1%', '5%'], y=[lower_bound_1, lower_bound_5], mode='lines+markers', marker=dict(size=10, color='#3a506b'), line=dict(width=2, color='#3a506b'), name='الحد الأدنى'))
		fig.add_trace(go.Scatter(x=['1%', '5%'], y=[upper_bound_1, upper_bound_5], mode='lines+markers', marker=dict(size=10, color='#30475e'), line=dict(width=2, color='#30475e'), name='الحد الأعلى'))
		fig.add_shape(type='rect', x0=-0.5, y0=0, x1=2.5, y1=lower_bound_1, fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0), layer='below')
		fig.add_shape(type='rect', x0=-0.5, y0=upper_bound_1, x1=2.5, y1=7, fillcolor='rgba(0, 255, 0, 0.1)', line=dict(width=0), layer='below')
		fig.add_shape(type='rect', x0=-0.5, y0=lower_bound_1, x1=2.5, y1=upper_bound_1, fillcolor='rgba(255, 255, 0, 0.1)', line=dict(width=0), layer='below')
		fig.update_layout(title="مثال على اختبار الحدود (Bounds Test)", xaxis_title="مستويات المعنوية", yaxis_title="قيمة إحصائية F", height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
		st.plotly_chart(fig, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق نموذج ARDL")
		code = """
        from statsmodels.tsa.api import ardl_select_order, ARDL
        import pandas as pd

        # Load data
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # Select optimal lag order
        sel_order = ardl_select_order(
            endog=df['y'], exog=df[['x1', 'x2']], maxlag=4, trend='c'
        )
        
        # Fit ARDL model
        model = sel_order.model.fit()

        # Print model summary
        print(model.summary())

        # Perform bounds test
        bounds_test = model.bounds_test()
        print(bounds_test)

        # Get long-run coefficients
        long_run = model.long_run_effects()
        print(long_run)
        """
		st.code(code, language='python')

# NARDL Model
elif choice == "نموذج NARDL":
	st.header("نموذج NARDL (Nonlinear ARDL)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة التأثيرات الديناميكية غير المتماثلة للمتغيرات المستقلة على المتغير التابع في الأجل الطويل والقصير.

        ### الشروط الأساسية
        - نفس الشروط المتعلقة بنموذج ARDL فيما يتعلق بالاستقرارية (I(0) أو I(1) وليس I(2))
        - يمكن أن يكون هناك feedback بين المتغيرات المستقلة والمتغير التابع

        ### حالات خاصة وتعديلات
        - في وجود مشكل singularity، يمكن الانتقال إلى طريقة التقدير بالخطوتين (two-step)
        - في حالة سيطرة تأثيرات موجبة على التأثيرات السالبة أو العكس، يمكن اللجوء إلى نماذج Multiple or Threshold ARDL
        
        ### الصيغة الرياضية
        """)
		st.latex(r"""
        \Delta y_t = \alpha_0 + \delta y_{t-1} + \theta^+ x^+_{t-1} + \theta^- x^-_{t-1} + \dots + \varepsilon_t
        """)
		st.markdown("حيث:")
		st.latex(r"""
        \begin{align}
        x_t^+ &= \sum_{j=1}^{t} \Delta x_j^+ = \sum_{j=1}^{t} \max(\Delta x_j, 0) \\
        x_t^- &= \sum_{j=1}^{t} \Delta x_j^- = \sum_{j=1}^{t} \min(\Delta x_j, 0)
        \end{align}
        """)
		st.markdown("""
        ### اختبار عدم التماثل
        بعد تقدير نموذج NARDL، يتم اختبار وجود تأثيرات غير متماثلة باستخدام اختبار Wald على المعاملات.
        - **عدم تماثل طويل الأجل:** $\theta^+ = \theta^-$
        - **عدم تماثل قصير الأجل:** $\beta_j^+ = \beta_j^-$
        
        ### مراحل تطبيق نموذج NARDL
        1. التأكد من استقرارية المتغيرات (I(0) أو I(1) وليس I(2))
        2. تفكيك المتغيرات المستقلة إلى مكونات موجبة وسالبة
        3. تقدير نموذج NARDL وإجراء اختبار الحدود
        4. اختبار عدم التماثل في الأجلين الطويل والقصير
        5. تحليل المعاملات وتفسير النتائج
        """)

	with col2:
		# Visualization 1
		np.random.seed(42)
		t = np.linspace(0, 10, 100)
		x = np.sin(t) + 0.1 * t
		dx = np.diff(x, prepend=x[0])
		x_pos = np.maximum(dx, 0).cumsum()
		x_neg = np.minimum(dx, 0).cumsum()
		fig1 = go.Figure()
		fig1.add_trace(go.Scatter(x=t, y=x, mode='lines', name='المتغير المستقل (x)', line=dict(color='#3a506b', width=2)))
		fig1.add_trace(go.Scatter(x=t, y=x_pos, mode='lines', name='المكون الموجب (x⁺)', line=dict(color='#2ecc71', width=2)))
		fig1.add_trace(go.Scatter(x=t, y=x_neg, mode='lines', name='المكون السالب (x⁻)', line=dict(color='#e74c3c', width=2)))
		fig1.update_layout(title="تفكيك المتغير المستقل في نموذج NARDL", xaxis_title="الزمن", yaxis_title="القيمة", height=300)
		st.plotly_chart(fig1, use_container_width=True)

		# Visualization 2
		t_sim = np.arange(20)
		cum_effect_pos = np.concatenate([np.zeros(5), 0.5 + 0.1 * np.arange(15)])
		cum_effect_neg = np.concatenate([np.zeros(10), -1 - 0.2 * np.arange(10)])
		fig2 = go.Figure()
		fig2.add_trace(go.Scatter(x=t_sim, y=cum_effect_pos, mode='lines', name='التأثير التراكمي للصدمة الإيجابية', line=dict(color='#2ecc71', width=2, dash='dash')))
		fig2.add_trace(go.Scatter(x=t_sim, y=cum_effect_neg, mode='lines', name='التأثير التراكمي للصدمة السلبية', line=dict(color='#e74c3c', width=2, dash='dash')))
		fig2.update_layout(title="التأثيرات التراكمية غير المتماثلة للصدمات", xaxis_title="الفترات الزمنية", yaxis_title="التأثير", height=300)
		st.plotly_chart(fig2, use_container_width=True)
		
		# Code Example (Conceptual)
		st.markdown("### مثال تطبيقي (مفاهيمي)")
		st.info("لا توجد حزمة بايثون قياسية لـ NARDL، لذلك الكود التالي هو للتوضيح المفاهيمي.")
		code = """
        import pandas as pd
        import statsmodels.api as sm

        # Load data
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # 1. Decompose independent variable 'x'
        df['dx'] = df['x'].diff().fillna(0)
        df['x_pos'] = df['dx'][df['dx'] > 0].cumsum().fillna(0)
        df['x_neg'] = df['dx'][df['dx'] < 0].cumsum().fillna(0)

        # 2. Define model variables (conceptual)
        # y ~ y_lag + x_pos + x_neg + dy_lag + dx_pos + dx_neg...
        
        # 3. Estimate with OLS (as ARDL)
        # model = sm.OLS(y, X).fit()
        
        # 4. Perform Wald test for asymmetry
        # H0: coef(x_pos) = coef(x_neg)
        """
		st.code(code, language='python')

# Dynamic Panel Models
elif choice == "نماذج البانل الديناميكية":
	st.header("نماذج البانل الديناميكية (Dynamic Panel Models)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        فهم وتقدير العلاقة الديناميكية بين المتغيرات لفهم سلوك المتغيرات عبر الزمن، سواء في إطار معادلة واحدة أو نظام من المعادلات.

        ### الشروط الأساسية للتقدير بطريقة GMM
        - يفترض أن المعامل المرتبط بالمتغير التابع يجب أن يكون أصغر من 1
        - من المستحسن أن لا يكون هناك cross-sectional dependence
        - يجب أن تكون شروط العزوم معرفة (شرط نظري)
        - يجب أن تكون instruments ليست كثيرة جداً وتكون معرفة ومحددة بشكل جيد حسب اختبارات Sargan و Hansen
        - في حالة المعامل المرتبط بالمتغير التابع المؤخر مساوي إلى الواحد، يمكن اللجوء إلى differenced GMM

        ### الصيغة الرياضية للنموذج الديناميكي البسيط
        """)
		st.latex(r"""
        y_{it} = \gamma y_{i,t-1} + \boldsymbol{x}_{it}' \boldsymbol{\beta} + \alpha_i + \varepsilon_{it}
        """)
		st.markdown("""
        ### نماذج البانل الديناميكية حسب أبعاد العينة
        - **N كبير، T صغير:**
            - طريقة Arellano-Bond (Difference GMM)
            - طريقة Arellano-Bover/Blundell-Bond (System GMM)
        - **N و T كبيران:**
            - طريقة Mean Group (MG)
            - طريقة Pooled Mean Group (PMG)
            - طريقة Dynamic Fixed Effects (DFE)

        ### الاختبارات المسبقة المهمة
        - اختبارات عدم تجانس الميول
        - اختبارات cross-sectional dependence
        - اختبارات الاستقرارية والتكامل المشترك للبانل
        """)

	with col2:
		# Visualization
		fig = go.Figure()
		true_gamma = 0.7
		t_values = [5, 10, 15, 20, 25, 30]
		gamma_ols = [0.9, 0.85, 0.82, 0.79, 0.77, 0.76]
		gamma_fe = [0.55, 0.58, 0.61, 0.63, 0.65, 0.66]
		gamma_gmm = [0.72, 0.71, 0.71, 0.7, 0.7, 0.7]
		fig.add_shape(type='line', x0=0, y0=true_gamma, x1=35, y1=true_gamma, line=dict(color='#2ecc71', width=2, dash='dash'))
		fig.add_trace(go.Scatter(x=t_values, y=gamma_ols, mode='lines+markers', name='تقدير OLS', line=dict(color='#e74c3c', width=2)))
		fig.add_trace(go.Scatter(x=t_values, y=gamma_fe, mode='lines+markers', name='تقدير Fixed Effects', line=dict(color='#3498db', width=2)))
		fig.add_trace(go.Scatter(x=t_values, y=gamma_gmm, mode='lines+markers', name='تقدير GMM', line=dict(color='#f39c12', width=2)))
		fig.add_annotation(x=30, y=true_gamma, text="القيمة الحقيقية", showarrow=False, yshift=10)
		fig.update_layout(title="تحيز التقدير في النماذج الديناميكية حسب T", xaxis_title="عدد الفترات الزمنية (T)", yaxis_title="تقدير المعامل γ", height=350)
		st.plotly_chart(fig, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق نموذج بانل ديناميكي (System GMM)")
		code = """
        from linearmodels.panel import PanelOLS, RandomEffects, PanelGMM
        import pandas as pd
        import statsmodels.api as sm

        # Load data and set index
        df = pd.read_csv('data.csv')
        df = df.set_index(['id', 'time'])

        # Define model formula
        formula = 'y ~ 1 + y_lag1 + x1 + x2'

        # Estimate System GMM
        # Note: Instruments need to be carefully chosen.
        # This is a simplified example.
        model = PanelGMM.from_formula(formula=formula, data=df)
        results = model.fit()

        # Print summary
        print(results)

        # Check instrument validity
        print(results.sargan)
        """
		st.code(code, language='python')

# Static Panel Models
elif choice == "نماذج البانل الساكنة":
	st.header("نماذج البانل الساكنة (Static Panel Models)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة التأثيرات الساكنة للمتغيرات المستقلة على المتغير التابع، مع التحكم في الخصائص غير المشاهدة للوحدات.

        ### الشروط الأساسية
        - نماذج البانل الساكنة التقليدية تفترض أن الميول (slopes) ثابتة.
        - الاختيار بين التأثيرات الثابتة والعشوائية يعتمد على اختبار Hausman.

        ### أنواع وتعديلات النموذج
        - **Fixed Effects (FE):** يتحكم في الخصائص الثابتة عبر الزمن لكل وحدة.
        - **Random Effects (RE):** يفترض أن التأثيرات غير مرتبطة بالمتغيرات المستقلة.
        - **Pooled OLS:** يتجاهل بنية البانل ويعتبر البيانات مقطعية.
        
        ### الصيغة الرياضية
        """)
		st.markdown("#### نموذج التأثيرات الثابتة (Fixed Effects Model)")
		st.latex(r"y_{it} = \boldsymbol{x}_{it}' \boldsymbol{\beta} + \alpha_i + \varepsilon_{it}")
		st.markdown("#### نموذج التأثيرات العشوائية (Random Effects Model)")
		st.latex(r"y_{it} = \boldsymbol{x}_{it}' \boldsymbol{\beta} + (\alpha + u_i) + \varepsilon_{it}")
		st.markdown("""
        ### الاختبارات المهمة
        - **اختبار Hausman:** للمفاضلة بين FE و RE.
        - **اختبار Breusch-Pagan LM:** للمفاضلة بين RE و Pooled OLS.
        - **اختبار F:** للمفاضلة بين FE و Pooled OLS.
        - **اختبارات Cross-sectional Dependence, Heteroskedasticity, Serial Correlation.**

        ### معالجة المشاكل
        - **Heteroskedasticity / Serial Correlation:** استخدام Robust/Clustered Standard Errors.
        - **Cross-sectional Dependence:** استخدام Driscoll-Kraay Standard Errors.
        - **Endogeneity:** استخدام Instrumental Variables (Panel IV).
        """)

	with col2:
		# Visualization
		fig1 = go.Figure()
		np.random.seed(42)
		x = np.linspace(0, 10, 20)
		y1 = 2 + 1.5 * x + np.random.normal(0, 1, 20)
		y2 = 2 + 1.5 * x + np.random.normal(0, 1, 20)
		all_x = np.concatenate([x, x])
		all_y = np.concatenate([y1, y2])
		coef = np.polyfit(all_x, all_y, 1)
		line = coef[0] * np.linspace(0, 10, 100) + coef[1]
		fig1.add_trace(go.Scatter(x=x, y=y1, mode='markers', name='المجموعة 1', marker=dict(color='#3a506b')))
		fig1.add_trace(go.Scatter(x=x, y=y2, mode='markers', name='المجموعة 2', marker=dict(color='#f05454')))
		fig1.add_trace(go.Scatter(x=np.linspace(0, 10, 100), y=line, mode='lines', name='خط الانحدار المجمع', line=dict(color='black')))
		fig1.update_layout(title="نموذج الانحدار التجميعي (Pooled OLS)", height=250, showlegend=False)
		st.plotly_chart(fig1, use_container_width=True)

		fig2 = go.Figure()
		y_fe_1 = 1 + 1.5 * x + np.random.normal(0, 0.7, 20)
		y_fe_2 = 5 + 1.5 * x + np.random.normal(0, 0.7, 20)
		fig2.add_trace(go.Scatter(x=x, y=y_fe_1, mode='markers', name='المجموعة 1', marker=dict(color='#3a506b')))
		fig2.add_trace(go.Scatter(x=x, y=y_fe_2, mode='markers', name='المجموعة 2', marker=dict(color='#f05454')))
		fig2.add_trace(go.Scatter(x=x, y=1 + 1.5 * x, mode='lines', line=dict(color='#3a506b')))
		fig2.add_trace(go.Scatter(x=x, y=5 + 1.5 * x, mode='lines', line=dict(color='#f05454')))
		fig2.update_layout(title="نموذج التأثيرات الثابتة (Fixed Effects)", height=250, showlegend=False)
		st.plotly_chart(fig2, use_container_width=True)

		# Code Example
		st.markdown("### مثال على تطبيق نماذج البانل الساكنة")
		code = """
        from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
        import pandas as pd

        # Load data and set index
        df = pd.read_csv('data.csv')
        df = df.set_index(['id', 'time'])

        # Pooled OLS
        pooled_model = PooledOLS.from_formula('y ~ 1 + x1 + x2', data=df)
        pooled_results = pooled_model.fit()

        # Fixed Effects
        fe_model = PanelOLS.from_formula('y ~ 1 + x1 + x2 + EntityEffects', data=df)
        fe_results = fe_model.fit()

        # Random Effects
        re_model = RandomEffects.from_formula('y ~ 1 + x1 + x2', data=df)
        re_results = re_model.fit()

        # Hausman Test (conceptual)
        # Compare coefficients of fe_results and re_results
        # No direct function in linearmodels, needs manual implementation or statsmodels
        print(fe_results)
        print(re_results)
        """
		st.code(code, language='python')

# Contradictions in Arab Studies
elif choice == "المتناقضات في الدراسات العربية":
	st.header("المتناقضات في الدراسات العربية")

	st.error("""
    ### أهم المتناقضات المنهجية الشائعة
    - **استخدام اختبار جوهانسون مع تغيرات هيكلية:** اختبار جوهانسون القياسي يفترض عدم وجود كسور هيكلية.
    - **الجمع بين ARDL و VAR:** الأول أحادي المعادلة (single-equation) والآخر نظام معادلات (system)، ولهما افتراضات مختلفة حول الداخلية (endogeneity).
    - **الجمع بين اختبار جوهانسون و Bounds Test:** الأول يختبر التكامل المشترك في نظام (علاقة تبادلية) والثاني في معادلة واحدة (علاقة أحادية الاتجاه).
    - **الجمع بين اختبارات الجيل الأول والثاني للبانل:** يجب اختيار الجيل المناسب بناءً على وجود أو غياب الاعتماد المقطعي (cross-sectional dependence).
    - **الجمع بين ARDL-PMG و ARDL-CS:** يجب اختيار النموذج بناءً على وجود أو غياب الاعتماد المقطعي.
    """)

	# Visualization
	fig = go.Figure()
	contradictions = ["جوهانسون + تغير هيكلي", "ARDL + VAR", "جوهانسون + Bounds Test", "جيل أول + ثاني للبانل", "PMG + CS-ARDL"]
	frequency = [68, 45, 72, 53, 40]
	fig.add_trace(go.Bar(x=contradictions, y=frequency, marker_color='#c0392b'))
	fig.update_layout(title="تكرار المتناقضات المنهجية في الدراسات", yaxis_title="تكرار الظهور (تقديري)")
	st.plotly_chart(fig, use_container_width=True)

	st.success("""
    ### نصائح لتجنب المتناقضات
    1. **فهم أساسيات النموذج:** فهم الافتراضات الأساسية والشروط اللازمة لكل نموذج.
    2. **اختيار النموذج المناسب:** يجب أن يتناسب النموذج مع طبيعة البيانات وأهداف الدراسة.
    3. **إجراء الاختبارات التشخيصية:** التحقق من صلاحية النموذج والاختبارات المسبقة (مثل الاعتماد المقطعي).
    4. **مراعاة خصائص البيانات:** الانتباه إلى خصائص البيانات مثل الاستقرارية والتغيرات الهيكلية.
    5. **تجنب الجمع بين النماذج المتعارضة:** لا تستخدم نماذج ذات افتراضات متعارضة في نفس الدراسة لنفس الهدف.
    """)

# General Notes
elif choice == "ملاحظات عامة":
	st.header("ملاحظات عامة")

	st.markdown("""
    - هذا المخطط يركز على أهم النماذج **التفسيرية** في الدراسات العربية، وليست التنبؤية.
    - الشروط المذكورة هي بصفة عامة، وكل نموذج له شروط مفصلة واختبارات تشخيصية خاصة به.
    - خاصية مشتركة بين كل النماذج هي ضرورة أن تكون البواقي خالية من المشاكل (ارتباط ذاتي، عدم تجانس التباين).
    - المعيار الأهم لاختيار نموذج معين هو مدى توافق أهدافه مع إشكالية البحث.
    - لتعلم أي نموذج، يجب التركيز على أهدافه، شروطه، وكيفية تطبيقه وتفسير نتائجه.
    """)

	# Visualization
	fig = go.Figure()
	criteria = ["توافق النموذج مع أهداف الدراسة", "قدرة النموذج على الإجابة عن إشكالية البحث", "تحقق شروط تطبيق النموذج", "توافر البيانات اللازمة", "سهولة التفسير والتحليل"]
	importance = [5, 4.8, 4.2, 3.5, 3.2]
	fig.add_trace(go.Bar(x=importance, y=criteria, orientation='h', marker=dict(color='#3a506b')))
	fig.update_layout(title="معايير اختيار النموذج القياسي المناسب (حسب الأهمية)", xaxis_title="درجة الأهمية", yaxis=dict(autorange="reversed"))
	st.plotly_chart(fig, use_container_width=True)

	# Recommendations
	st.info("""
    ### توصيات لاستخدام النماذج القياسية
    1. ضرورة فهم الأسس النظرية والافتراضات الأساسية للنماذج قبل تطبيقها.
    2. أهمية اختيار النموذج المناسب وفقاً لطبيعة البيانات وأهداف الدراسة.
    3. ضرورة إجراء الاختبارات التشخيصية للتحقق من صلاحية النموذج.
    4. تجنب استخدام النماذج ذات الافتراضات المتناقضة في نفس الدراسة.
    5. الاطلاع المستمر على التطورات الحديثة في مجال النمذجة القياسية.
    """)
	st.info("إعداد: Merwan Roudane")

st.markdown("---")
st.markdown("© 2025 - النماذج القياسية الأكثر شهرة عربياً")
