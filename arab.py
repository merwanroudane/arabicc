import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'

# تعيين صفحة Streamlit
st.set_page_config(
	page_title="النماذج القياسية الأكثر شهرة عربيا",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded"
)

# تعريف CSS للعناصر العربية
# The custom CSS block has been removed as per the request to remove HTML tags.
# The app will now use Streamlit's default styling.
st.markdown("""
<style>
    /* Custom CSS is removed, but we keep basic RTL direction for layout */
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


# العنوان الرئيسي
st.markdown("# النماذج القياسية الأكثر شهرة عربياً")
st.markdown("### إعداد: Merwan Roudane")

# إضافة شريط جانبي
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

# إضافة معلومات إضافية في الشريط الجانبي
st.sidebar.markdown("---")
st.sidebar.info("هذا المخطط يتناول أهم النماذج التفسيرية في الدراسات العربية وليست التنبؤية")
st.sidebar.info("الشروط المذكورة هي بصفة عامة وليست مفصلة حيث تحتاج الشروط المفصلة إلى مخطط لكل نموذج على حدى")


# دالة لإنشاء رسم بياني للنماذج
def create_model_tree():
	fig = go.Figure()

	models = [
		"نماذج الانحدار الخطي", "نماذج الانحدار الكمي", "نماذج المعادلات الآنية",
		"نموذج VAR", "نموذج VECM", "نموذج ARDL", "نموذج NARDL",
		"نماذج البانل الديناميكية", "نماذج البانل الساكنة"
	]

	x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	y = [3, 2, 3, 1, 1, 2, 2, 3, 3]

	# إضافة النقاط
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode='markers+text',
		marker=dict(size=20, color=['#3a506b'] * len(models)),
		text=models,
		textposition="top center",
		textfont=dict(size=14, color='black', family='Arial'),
		hoverinfo='text'
	))

	# إضافة الخطوط للربط
	fig.add_shape(type="line", x0=1, y0=3, x1=3, y1=3, line=dict(color="#718096", width=2))
	fig.add_shape(type="line", x0=4, y0=1, x1=7, y1=1, line=dict(color="#718096", width=2))
	fig.add_shape(type="line", x0=8, y0=3, x1=9, y1=3, line=dict(color="#718096", width=2))

	# تنسيق الرسم البياني
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


# إنشاء رسم بياني لمقارنة شروط النماذج
def create_conditions_comparison():
	categories = ['استقرارية البيانات', 'حجم العينة', 'التوزيع الطبيعي', 'مشاكل التوصيف', 'العلاقة السببية']

	models = ['ARDL', 'VAR', 'VECM', 'نماذج البانل']
	values = [
		[3, 3, 2, 4, 5],  # ARDL
		[5, 4, 3, 3, 5],  # VAR
		[5, 4, 2, 3, 5],  # VECM
		[4, 5, 2, 4, 3],  # نماذج البانل
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


# الصفحة الرئيسية
if choice == "الرئيسية":
	st.markdown("## مقدمة عن النماذج القياسية الشائعة الاستخدام عربياً")

	st.info(
		"تقدم هذه الوثيقة عرضاً للنماذج القياسية الأكثر شيوعاً في الدراسات العربية مع توضيح الشروط الأساسية لاستخدامها. تشمل هذه النماذج أنواعاً مختلفة من تحليل الانحدار، ونماذج المعادلات الآنية، ونماذج السلاسل الزمنية، ونماذج البانل."
	)

	# عرض الرسم البياني للنماذج
	st.plotly_chart(create_model_tree(), use_container_width=True)

	# عرض مقارنة شروط النماذج
	st.plotly_chart(create_conditions_comparison(), use_container_width=True)

	# معلومات إضافية
	st.markdown("""
    ### أهمية اختيار النموذج المناسب
    يعتمد اختيار النموذج المناسب على عدة عوامل أهمها:
    - هدف الدراسة (تفسيري أم تنبؤي)
    - طبيعة البيانات (مقطعية، سلاسل زمنية، بيانات بانل)
    - خصائص المتغيرات (استقرارية، توزيع، إلخ)
    - العلاقة بين المتغيرات (أحادية الاتجاه، تبادلية)
    """)

# نموذج الانحدار الخطي
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
		# رسم بياني للتوضيح
		fig = go.Figure()

		# إنشاء بيانات وهمية للتوضيح
		np.random.seed(42)
		x = np.linspace(0, 10, 100)
		y = 2 * x + 1 + np.random.normal(0, 2, 100)

		# إضافة نقاط البيانات
		fig.add_trace(go.Scatter(
			x=x, y=y,
			mode='markers',
			name='البيانات',
			marker=dict(color='#3a506b', size=8)
		))

		# إضافة خط الانحدار
		coef = np.polyfit(x, y, 1)
		line = coef[0] * x + coef[1]
		fig.add_trace(go.Scatter(
			x=x, y=line,
			mode='lines',
			name='خط الانحدار',
			line=dict(color='#f05454', width=3)
		))

		fig.update_layout(
			title="مثال على الانحدار الخطي البسيط",
			xaxis_title="المتغير المستقل",
			yaxis_title="المتغير التابع",
			legend_title="البيانات",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال لنموذج انحدار متعدد
		st.markdown("### مثال على بنية نموذج الانحدار المتعدد")
		code = """
        import statsmodels.api as sm
        import pandas as pd

        # إعداد البيانات
        df = pd.read_csv('data.csv')

        # تحديد المتغيرات المستقلة والتابعة
        X = df[['x1', 'x2', 'x3']]
        X = sm.add_constant(X)
        y = df['y']

        # تقدير النموذج
        model = sm.OLS(y, X).fit()

        # عرض النتائج
        print(model.summary())
        """
		st.code(code, language='python')

# نموذج الانحدار الكمي
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
        - Quantile in Quantile Regression: نموذج أكثر مرونة يسمح بدراسة العلاقة بين الكميات للمتغيرات المستقلة والتابعة

        ### ميزات استخدام الانحدار الكمي
        - أقل تأثراً بالقيم المتطرفة مقارنة بالانحدار العادي
        - يسمح بتحليل تأثير المتغيرات المستقلة على كامل توزيع المتغير التابع وليس فقط على متوسطه
        - لا يتطلب افتراضات قوية حول توزيع البواقي
        """)

	with col2:
		# رسم بياني للتوضيح
		np.random.seed(42)
		x = np.linspace(0, 10, 200)
		# إنشاء بيانات ذات توزيع غير متماثل
		y = 2 * x + 1 + np.random.exponential(scale=2, size=200)

		# تقدير انحدار كمي (تقريبي للعرض فقط)
		q_25 = 2 * x + 0.2  # تقريب للكمية 0.25
		q_50 = 2 * x + 1  # تقريب للكمية 0.50 (الوسيط)
		q_75 = 2 * x + 2.5  # تقريب للكمية 0.75

		fig = go.Figure()

		# إضافة نقاط البيانات
		fig.add_trace(go.Scatter(
			x=x, y=y,
			mode='markers',
			name='البيانات',
			marker=dict(color='#3a506b', size=6, opacity=0.7)
		))

		# إضافة خطوط الانحدار الكمي
		fig.add_trace(go.Scatter(
			x=x, y=q_25,
			mode='lines',
			name='الكمية 0.25',
			line=dict(color='#f05454', width=2)
		))

		fig.add_trace(go.Scatter(
			x=x, y=q_50,
			mode='lines',
			name='الكمية 0.50 (الوسيط)',
			line=dict(color='#30475e', width=2)
		))

		fig.add_trace(go.Scatter(
			x=x, y=q_75,
			mode='lines',
			name='الكمية 0.75',
			line=dict(color='#7b68ee', width=2)
		))

		fig.update_layout(
			title="مثال على الانحدار الكمي",
			xaxis_title="المتغير المستقل",
			yaxis_title="المتغير التابع",
			legend_title="البيانات والكميات",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# كود مثال
		st.markdown("### مثال على تطبيق الانحدار الكمي")
		code = """
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd

        # إعداد البيانات
        df = pd.read_csv('data.csv')

        # تقدير نموذج الانحدار الكمي عند كميات مختلفة
        q_25 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.25)
        q_50 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.50)
        q_75 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.75)

        # عرض النتائج
        print(q_50.summary())
        """
		st.code(code, language='python')

# نموذج المعادلات الآنية
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
		# رسم بياني توضيحي للعلاقات المتشابكة
		nodes = ['Y₁', 'Y₂', 'X₁', 'X₂']
		edges = [('Y₁', 'Y₂'), ('Y₂', 'Y₁'), ('X₁', 'Y₁'), ('X₁', 'Y₂'), ('X₂', 'Y₁'), ('X₂', 'Y₂')]

		# إنشاء الرسم البياني التوضيحي
		G = {node: [] for node in nodes}
		for edge in edges:
			G[edge[0]].append(edge[1])

		# تحديد مواقع النقاط
		pos = {
			'Y₁': [0, 0.5],
			'Y₂': [1, 0.5],
			'X₁': [0.25, 1],
			'X₂': [0.75, 1]
		}

		fig = go.Figure()

		# إضافة الحواف
		for source, targets in G.items():
			for target in targets:
				fig.add_trace(go.Scatter(
					x=[pos[source][0], pos[target][0]],
					y=[pos[source][1], pos[target][1]],
					mode='lines',
					line=dict(width=2, color='#718096'),
					hoverinfo='none'
				))

		# إضافة النقاط
		node_x = [pos[node][0] for node in nodes]
		node_y = [pos[node][1] for node in nodes]

		colors = ['#f05454', '#f05454', '#30475e', '#30475e']

		fig.add_trace(go.Scatter(
			x=node_x,
			y=node_y,
			mode='markers+text',
			marker=dict(
				size=30,
				color=colors,
				line=dict(width=2, color='white')
			),
			text=nodes,
			textposition="middle center",
			textfont=dict(size=20, color='white'),
			hoverinfo='text',
			hovertext=[
				"المتغير التابع في المعادلة الأولى",
				"المتغير التابع في المعادلة الثانية",
				"متغير مستقل خارجي",
				"متغير مستقل خارجي"
			]
		))

		fig.update_layout(
			title="العلاقات المتشابكة في نموذج المعادلات الآنية",
			showlegend=False,
			height=400,
			plot_bgcolor='#f9f9f9',
			xaxis=dict(
				showticklabels=False,
				showgrid=False,
				zeroline=False,
				range=[-0.1, 1.1]
			),
			yaxis=dict(
				showticklabels=False,
				showgrid=False,
				zeroline=False,
				range=[0.4, 1.1]
			)
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج المعادلات الآنية
		st.markdown("### مثال على تطبيق نموذج المعادلات الآنية")
		code = """
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        import pandas as pd

        # إعداد البيانات
        df = pd.read_csv('data.csv')

        # تعريف المتغيرات
        endog = df['y1']            # المتغير التابع في المعادلة الأولى
        exog = df[['const', 'y2']]  # المتغيرات المستقلة (بما فيها المتغير الداخلي)
        instruments = df[['const', 'x1', 'x2']]  # الأدوات (بما فيها المتغيرات الخارجية)

        # تقدير النموذج باستخدام طريقة 2SLS
        model = IV2SLS(endog, exog, instruments).fit()

        # عرض النتائج
        print(model.summary())
        """
		st.code(code, language='python')

# نموذج VAR
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
        \begin{pmatrix} y_{1t} \\ y_{2t} \\ \vdots \\ y_{nt} \end{pmatrix} = 
        \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix} +
        \begin{pmatrix} 
        \phi_{11}^1 & \phi_{12}^1 & \cdots & \phi_{1n}^1 \\
        \phi_{21}^1 & \phi_{22}^1 & \cdots & \phi_{2n}^1 \\
        \vdots & \vdots & \ddots & \vdots \\
        \phi_{n1}^1 & \phi_{n2}^1 & \cdots & \phi_{nn}^1
        \end{pmatrix}
        \begin{pmatrix} y_{1,t-1} \\ y_{2,t-1} \\ \vdots \\ y_{n,t-1} \end{pmatrix} + \cdots +
        \begin{pmatrix} 
        \phi_{11}^p & \phi_{12}^p & \cdots & \phi_{1n}^p \\
        \phi_{21}^p & \phi_{22}^p & \cdots & \phi_{2n}^p \\
        \vdots & \vdots & \ddots & \vdots \\
        \phi_{n1}^p & \phi_{n2}^p & \cdots & \phi_{nn}^p
        \end{pmatrix}
        \begin{pmatrix} y_{1,t-p} \\ y_{2,t-p} \\ \vdots \\ y_{n,t-p} \end{pmatrix} +
        \begin{pmatrix} \varepsilon_{1t} \\ \varepsilon_{2t} \\ \vdots \\ \varepsilon_{nt} \end{pmatrix}
        """)

		st.markdown("""
        ### استخدامات النموذج
        - التنبؤ بالقيم المستقبلية للمتغيرات
        - تحليل الصدمات وتأثيرها على المتغيرات
        - تحليل تفكيك التباين (Variance Decomposition)
        - تحليل دوال الاستجابة النبضية (Impulse Response Functions)
        """)

	with col2:
		# رسم بياني لدالة الاستجابة النبضية (IRF)
		fig = go.Figure()

		# إنشاء بيانات وهمية لدالة الاستجابة النبضية
		periods = list(range(11))
		irf_values = [0, 0.05, 0.1, 0.14, 0.16, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01]
		confidence_upper = [v + 0.05 for v in irf_values]
		confidence_lower = [max(0, v - 0.05) for v in irf_values]

		# إضافة منطقة فاصل الثقة
		fig.add_trace(go.Scatter(
			x=periods + periods[::-1],
			y=confidence_upper + confidence_lower[::-1],
			fill='toself',
			fillcolor='rgba(58, 80, 107, 0.2)',
			line=dict(color='rgba(255, 255, 255, 0)'),
			hoverinfo='skip',
			showlegend=False
		))

		# إضافة دالة الاستجابة النبضية
		fig.add_trace(go.Scatter(
			x=periods,
			y=irf_values,
			mode='lines+markers',
			line=dict(color='#3a506b', width=3),
			marker=dict(size=8),
			name='دالة الاستجابة النبضية'
		))

		# إضافة خط الصفر
		fig.add_shape(
			type='line',
			x0=0, y0=0,
			x1=10, y1=0,
			line=dict(color='#718096', width=1, dash='dash')
		)

		fig.update_layout(
			title="مثال على دالة الاستجابة النبضية (IRF)",
			xaxis_title="الفترات الزمنية",
			yaxis_title="استجابة المتغير",
			height=300
		)

		st.plotly_chart(fig, use_container_width=True)

		# رسم بياني لتفكيك التباين
		fig = go.Figure()

		# إنشاء بيانات وهمية لتفكيك التباين
		periods = list(range(1, 11))
		var1 = [100, 90, 80, 75, 70, 68, 65, 63, 60, 58]
		var2 = [0, 5, 10, 12, 15, 16, 18, 19, 21, 22]
		var3 = [0, 5, 10, 13, 15, 16, 17, 18, 19, 20]

		# إضافة المساهمات المختلفة
		fig.add_trace(go.Bar(
			x=periods,
			y=var1,
			name='المتغير 1',
			marker_color='#3a506b'
		))

		fig.add_trace(go.Bar(
			x=periods,
			y=var2,
			name='المتغير 2',
			marker_color='#f05454'
		))

		fig.add_trace(go.Bar(
			x=periods,
			y=var3,
			name='المتغير 3',
			marker_color='#30475e'
		))

		fig.update_layout(
			title="مثال على تفكيك التباين",
			xaxis_title="الفترات الزمنية",
			yaxis_title="نسبة المساهمة (%)",
			barmode='stack',
			height=300
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج VAR
		st.markdown("### مثال على تطبيق نموذج VAR")
		code = """
        import pandas as pd
        from statsmodels.tsa.api import VAR

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # تحديد عدد الفجوات الزمنية المثلى
        model = VAR(df)
        results = model.select_order(maxlags=10)

        # تقدير النموذج
        var_model = model.fit(results.aic)

        # التنبؤ
        forecast = var_model.forecast(df.values[-results.aic:], steps=5)

        # تحليل دوال الاستجابة النبضية
        irf = var_model.irf(10)
        irf.plot()

        # تحليل تفكيك التباين
        fevd = var_model.fevd(10)
        fevd.plot()
        """
		st.code(code, language='python')

# نموذج VECM
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
		# رسم بياني توضيحي لآلية عمل VECM
		np.random.seed(42)
		t = np.linspace(0, 10, 200)

		# إنشاء سلسلتين زمنيتين متكاملتين مشتركاً
		equilibrium = 2 * t
		y1 = equilibrium + np.random.normal(0, 1, 200)
		y2 = equilibrium + np.random.normal(0, 1, 200)

		# إضافة انحراف في نقطة معينة ثم تصحيح
		shock_point = 100
		y1[shock_point:shock_point + 30] += np.linspace(0, 5, 30)
		y1[shock_point + 30:] += 5 - 5 * np.exp(-0.1 * np.arange(70))

		fig = go.Figure()

		# إضافة السلاسل الزمنية
		fig.add_trace(go.Scatter(
			x=t, y=y1,
			mode='lines',
			name='السلسلة الزمنية 1',
			line=dict(color='#3a506b', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t, y=y2,
			mode='lines',
			name='السلسلة الزمنية 2',
			line=dict(color='#f05454', width=2)
		))

		# إضافة التوازن طويل الأجل
		fig.add_trace(go.Scatter(
			x=t, y=equilibrium,
			mode='lines',
			name='التوازن طويل الأجل',
			line=dict(color='#30475e', width=2, dash='dash')
		))

		# إشارة إلى نقطة الصدمة
		fig.add_annotation(
			x=t[shock_point], y=y1[shock_point],
			text="الصدمة",
			showarrow=True,
			arrowhead=1,
			ax=0, ay=-40
		)

		# إشارة إلى عملية التصحيح
		fig.add_annotation(
			x=t[shock_point + 50], y=y1[shock_point + 50],
			text="تصحيح الخطأ",
			showarrow=True,
			arrowhead=1,
			ax=0, ay=-40
		)

		fig.update_layout(
			title="آلية عمل نموذج تصحيح الخطأ (VECM)",
			xaxis_title="الزمن",
			yaxis_title="القيمة",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج VECM
		st.markdown("### مثال على تطبيق نموذج VECM")
		code = """
        import pandas as pd
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.vector_ar.vecm import VECM
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # اختبار التكامل المشترك
        johansen_test = coint_johansen(df, 0, 2)

        # تحديد عدد علاقات التكامل المشترك
        trace_stat = johansen_test.lr1
        trace_crit = johansen_test.cvt
        r = sum(trace_stat > trace_crit[:, 1])

        # تقدير نموذج VECM
        model = VECM(df, k_ar_diff=2, coint_rank=r, deterministic='ci')
        results = model.fit()

        # عرض النتائج
        print(results.summary())

        # استخراج معاملات تصحيح الخطأ
        alpha = results.alpha
        print("معاملات تصحيح الخطأ:")
        print(alpha)
        """
		st.code(code, language='python')

# نموذج ARDL
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
        \begin{align}
        \Delta y_t &= \alpha_0 + \alpha_1 t + \delta_1 y_{t-1} + \delta_2 x_{t-1} + \delta_3 z_{t-1} + ... \\
        &+ \sum_{i=1}^{p} \beta_i \Delta y_{t-i} + \sum_{i=0}^{q} \gamma_i \Delta x_{t-i} + \sum_{i=0}^{r} \theta_i \Delta z_{t-i} + ... + \varepsilon_t
        \end{align}
        """)

		st.markdown("""
        ### مزايا نموذج ARDL
        - يمكن استخدامه مع متغيرات ذات درجات تكامل مختلفة (I(0) و I(1) ولكن ليس I(2))
        - يسمح بتقدير العلاقات طويلة وقصيرة الأجل في معادلة واحدة
        - يعالج مشكلة Endogeneity وارتباط البواقي من خلال إدراج عدد كافٍ من الفجوات الزمنية
        - يمكن استخدامه مع عينات صغيرة نسبياً

        ### اختبارات الحدود (Bounds Test)
        يستخدم اختبار الحدود ARDL Bounds Test للتحقق من وجود علاقة توازن طويلة الأجل بين المتغيرات، بغض النظر عن كونها I(0) أو I(1).

        الفرضية الصفرية: لا توجد علاقة تكامل (توازن) طويلة الأجل.
        
        الفرضية البديلة: توجد علاقة تكامل طويلة الأجل.

        ### مراحل تطبيق نموذج ARDL
        1. التأكد من استقرارية المتغيرات (I(0) أو I(1) وليس I(2))
        2. تحديد العدد الأمثل للفجوات الزمنية باستخدام معايير المعلومات
        3. تقدير نموذج ARDL
        4. إجراء اختبار الحدود Bounds Test للتحقق من وجود علاقة توازن طويلة الأجل
        5. تقدير العلاقة طويلة الأجل ونموذج تصحيح الخطأ
        6. إجراء اختبارات التشخيص للتحقق من صلاحية النموذج
        """)

	with col2:
		# رسم بياني توضيحي لاختبار الحدود
		fig = go.Figure()

		# إنشاء بيانات وهمية
		f_stat = 5.2
		lower_bound_1 = 2.8
		upper_bound_1 = 3.8
		lower_bound_5 = 2.1
		upper_bound_5 = 3.0
		lower_bound_10 = 1.8
		upper_bound_10 = 2.7

		# إضافة القيمة المحسوبة لإحصائية F
		fig.add_trace(go.Scatter(
			x=['القيمة المحسوبة'],
			y=[f_stat],
			mode='markers',
			marker=dict(size=15, color='#f05454'),
			name='إحصائية F المحسوبة'
		))

		# إضافة حدود الاختبار
		fig.add_trace(go.Scatter(
			x=['1%', '5%', '10%'],
			y=[lower_bound_1, lower_bound_5, lower_bound_10],
			mode='lines+markers',
			marker=dict(size=10, color='#3a506b'),
			line=dict(width=2, color='#3a506b'),
			name='الحد الأدنى'
		))

		fig.add_trace(go.Scatter(
			x=['1%', '5%', '10%'],
			y=[upper_bound_1, upper_bound_5, upper_bound_10],
			mode='lines+markers',
			marker=dict(size=10, color='#30475e'),
			line=dict(width=2, color='#30475e'),
			name='الحد الأعلى'
		))

		# تحديد المناطق
		fig.add_shape(
			type='rect',
			x0=-0.5, y0=0,
			x1=3.5, y1=lower_bound_1,
			fillcolor='rgba(255, 0, 0, 0.1)',
			line=dict(width=0),
			layer='below'
		)

		fig.add_shape(
			type='rect',
			x0=-0.5, y0=upper_bound_1,
			x1=3.5, y1=7,
			fillcolor='rgba(0, 255, 0, 0.1)',
			line=dict(width=0),
			layer='below'
		)

		fig.add_shape(
			type='rect',
			x0=-0.5, y0=lower_bound_1,
			x1=3.5, y1=upper_bound_1,
			fillcolor='rgba(255, 255, 0, 0.1)',
			line=dict(width=0),
			layer='below'
		)

		fig.update_layout(
			title="مثال على اختبار الحدود (Bounds Test)",
			xaxis_title="مستويات المعنوية",
			yaxis_title="قيمة إحصائية F",
			height=300,
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=1.02,
				xanchor="right",
				x=1
			)
		)

		# إضافة تفسير المناطق
		fig.add_annotation(
			x=2.5, y=6.5,
			text="منطقة رفض الفرضية الصفرية<br>(وجود علاقة تكامل مشترك)",
			showarrow=False,
			bgcolor='rgba(0, 255, 0, 0.1)',
			bordercolor='rgba(0, 255, 0, 0.5)',
			borderwidth=1,
			borderpad=4,
			font=dict(size=10)
		)

		fig.add_annotation(
			x=2.5, y=1,
			text="منطقة قبول الفرضية الصفرية<br>(عدم وجود علاقة تكامل مشترك)",
			showarrow=False,
			bgcolor='rgba(255, 0, 0, 0.1)',
			bordercolor='rgba(255, 0, 0, 0.5)',
			borderwidth=1,
			borderpad=4,
			font=dict(size=10)
		)

		fig.add_annotation(
			x=2.5, y=3.3,
			text="منطقة غير حاسمة",
			showarrow=False,
			bgcolor='rgba(255, 255, 0, 0.1)',
			bordercolor='rgba(255, 255, 0, 0.5)',
			borderwidth=1,
			borderpad=4,
			font=dict(size=10)
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج ARDL
		st.markdown("### مثال على تطبيق نموذج ARDL")
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.tsa.ardl import ardl_select_order, ARDL

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # تحديد العدد الأمثل للفجوات الزمنية
        order_select = ardl_select_order(
            endog=df['y'],
            exog=df[['x1', 'x2']],
            maxlag=4,
            maxorder=4,
            trend='c',
            ic='aic'
        )

        # تقدير نموذج ARDL
        ardl_model = ARDL(
            endog=df['y'],
            exog=df[['x1', 'x2']],
            lags=order_select.lags,
            order=order_select.order,
            trend='c'
        )

        ardl_results = ardl_model.fit()
        print(ardl_results.summary())

        # إجراء اختبار الحدود
        bounds_test = ardl_results.bounds_test()
        print(bounds_test)

        # استخراج العلاقة طويلة الأجل
        long_run = ardl_results.long_run()
        print(long_run)
        """
		st.code(code, language='python')

# نموذج NARDL
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
        - في وجود مشكل singularity، يمكن الانتقال من طريقة التقدير بالخطوة الواحدة إلى طريقة التقدير بالخطوتين (two-step)
        - في حالة سيطرة تأثيرات موجبة على التأثيرات السالبة أو العكس، يمكن اللجوء إلى نماذج Multiple or Threshold ARDL
        - هناك نماذج أخرى غير شائعة في الأبحاث مثل Fuzzy ARDL أو Wavelet ARDL

        ### الصيغة الرياضية
        """)

		st.latex(r"""
        \begin{align}
        \Delta y_t &= \alpha_0 + \alpha_1 t + \delta_1 y_{t-1} + \delta_2^+ x^+_{t-1} + \delta_2^- x^-_{t-1} + \ldots \\
        &+ \sum_{i=1}^{p} \beta_i \Delta y_{t-i} + \sum_{i=0}^{q} (\gamma_i^+ \Delta x^+_{t-i} + \gamma_i^- \Delta x^-_{t-i}) + \ldots + \varepsilon_t
        \end{align}
        """)

		st.markdown("حيث:")
		st.latex(r"""
        \begin{align}
        x_t^+ &= \sum_{j=1}^{t} \Delta x_j^+ = \sum_{j=1}^{t} \max(\Delta x_j, 0) \\
        x_t^- &= \sum_{j=1}^{t} \Delta x_j^- = \sum_{j=1}^{t} \min(\Delta x_j, 0)
        \end{align}
        """)

		st.markdown("""
        ### الفرق بين ARDL و NARDL
        الفرق الرئيسي بين ARDL و NARDL هو أن NARDL يسمح بتأثيرات غير متماثلة للزيادات والانخفاضات في المتغيرات المستقلة. يتم تحقيق ذلك من خلال تفكيك المتغيرات المستقلة إلى مكونات موجبة وسالبة.

        ### اختبار عدم التماثل
        بعد تقدير نموذج NARDL، يمكن اختبار وجود تأثيرات غير متماثلة طويلة الأجل من خلال اختبار الفرضية:
        
        الفرضية الصفرية (تماثل طويل الأجل): $\frac{\delta_2^+}{-\delta_1} = \frac{\delta_2^-}{-\delta_1}$
        
        وبالمثل، يمكن اختبار عدم التماثل قصير الأجل من خلال اختبار الفرضية:
        
        الفرضية الصفرية (تماثل قصير الأجل): $\sum_{i=0}^{q} \gamma_i^+ = \sum_{i=0}^{q} \gamma_i^-$

        ### مراحل تطبيق نموذج NARDL
        1. التأكد من استقرارية المتغيرات (I(0) أو I(1) وليس I(2))
        2. تفكيك المتغيرات المستقلة إلى مكونات موجبة وسالبة
        3. تحديد العدد الأمثل للفجوات الزمنية
        4. تقدير نموذج NARDL
        5. إجراء اختبار الحدود للتحقق من وجود علاقة توازن طويلة الأجل
        6. اختبار عدم التماثل في الأجلين الطويل والقصير
        7. تحليل المعاملات وتفسير النتائج
        """)

	with col2:
		# رسم بياني توضيحي للتأثيرات غير المتماثلة
		np.random.seed(42)
		t = np.linspace(0, 10, 100)
		x = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, 100)

		# تفكيك المتغير إلى مكونات موجبة وسالبة
		dx = np.diff(x, prepend=x[0])
		dx_pos = np.maximum(dx, 0)
		dx_neg = np.minimum(dx, 0)

		x_pos = np.cumsum(dx_pos)
		x_neg = np.cumsum(dx_neg)

		# تأثيرات مختلفة للتغيرات الموجبة والسالبة
		y_pos_effect = 0.8 * x_pos
		y_neg_effect = 1.5 * x_neg

		# المتغير التابع النهائي
		y = y_pos_effect + y_neg_effect + np.random.normal(0, 0.2, 100)

		fig = go.Figure()

		# إضافة المتغير المستقل
		fig.add_trace(go.Scatter(
			x=t, y=x,
			mode='lines',
			name='المتغير المستقل (x)',
			line=dict(color='#3a506b', width=2)
		))

		# إضافة المكونات الموجبة والسالبة
		fig.add_trace(go.Scatter(
			x=t, y=x_pos,
			mode='lines',
			name='المكون الموجب (x⁺)',
			line=dict(color='#2ecc71', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t, y=x_neg,
			mode='lines',
			name='المكون السالب (x⁻)',
			line=dict(color='#e74c3c', width=2)
		))

		# إضافة المتغير التابع
		fig.add_trace(go.Scatter(
			x=t, y=y,
			mode='lines',
			name='المتغير التابع (y)',
			line=dict(color='#f05454', width=2)
		))

		fig.update_layout(
			title="تفكيك المتغير المستقل في نموذج NARDL",
			xaxis_title="الزمن",
			yaxis_title="القيمة",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# رسم بياني لتوضيح التأثيرات التراكمية غير المتماثلة
		t_sim = np.arange(20)

		# افتراض وجود صدمة إيجابية وصدمة سلبية
		shock_pos = np.zeros(20)
		shock_pos[5] = 1  # صدمة إيجابية في الفترة 5

		shock_neg = np.zeros(20)
		shock_neg[12] = -1  # صدمة سلبية في الفترة 12

		# التأثيرات التراكمية المختلفة
		cum_effect_pos = np.zeros(20)
		cum_effect_neg = np.zeros(20)

		for i in range(5, 20):
			if i == 5:
				cum_effect_pos[i] = 0.3
			elif i > 5 and i < 10:
				cum_effect_pos[i] = cum_effect_pos[i - 1] + 0.15 * (1 - cum_effect_pos[i - 1])
			else:
				cum_effect_pos[i] = cum_effect_pos[i - 1] + 0.05 * (0.8 - cum_effect_pos[i - 1])

		for i in range(12, 20):
			if i == 12:
				cum_effect_neg[i] = -0.5
			elif i > 12 and i < 15:
				cum_effect_neg[i] = cum_effect_neg[i - 1] - 0.2 * (-1.2 - cum_effect_neg[i - 1])
			else:
				cum_effect_neg[i] = cum_effect_neg[i - 1] - 0.1 * (-1.5 - cum_effect_neg[i - 1])

		fig2 = go.Figure()

		# إضافة الصدمات
		fig2.add_trace(go.Scatter(
			x=t_sim, y=shock_pos,
			mode='lines+markers',
			name='صدمة إيجابية',
			line=dict(color='#2ecc71', width=2)
		))

		fig2.add_trace(go.Scatter(
			x=t_sim, y=shock_neg,
			mode='lines+markers',
			name='صدمة سلبية',
			line=dict(color='#e74c3c', width=2)
		))

		# إضافة التأثيرات التراكمية
		fig2.add_trace(go.Scatter(
			x=t_sim, y=cum_effect_pos,
			mode='lines',
			name='التأثير التراكمي للصدمة الإيجابية',
			line=dict(color='#2ecc71', width=2, dash='dash')
		))

		fig2.add_trace(go.Scatter(
			x=t_sim, y=cum_effect_neg,
			mode='lines',
			name='التأثير التراكمي للصدمة السلبية',
			line=dict(color='#e74c3c', width=2, dash='dash')
		))

		fig2.update_layout(
			title="التأثيرات التراكمية غير المتماثلة للصدمات",
			xaxis_title="الفترات الزمنية",
			yaxis_title="التأثير",
			height=300
		)

		st.plotly_chart(fig2, use_container_width=True)

		# مثال على تطبيق نموذج NARDL
		st.markdown("### مثال على تطبيق نموذج NARDL")
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # تفكيك المتغير المستقل إلى مكونات موجبة وسالبة
        df['dx'] = df['x'].diff().fillna(0)
        df['dx_pos'] = df['dx'].apply(lambda x: max(x, 0))
        df['dx_neg'] = df['dx'].apply(lambda x: min(x, 0))

        df['x_pos'] = df['dx_pos'].cumsum()
        df['x_neg'] = df['dx_neg'].cumsum()

        # تقدير نموذج NARDL
        y = df['y']
        X = sm.add_constant(df[['y_lag1', 'x_pos_lag1', 'x_neg_lag1', 
                                'dy_lag1', 'dx_pos', 'dx_pos_lag1', 
                                'dx_neg', 'dx_neg_lag1']])

        model = sm.OLS(y, X).fit()
        print(model.summary())

        # اختبار التكامل المشترك (اختبار الحدود)
        # ...

        # اختبار عدم التماثل طويل الأجل
        # ...
        """
		st.code(code, language='python')

# نماذج البانل الديناميكية
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
        - يجب أن تكون instruments ليس كثيرة جداً وتكون معرفة ومحددة بشكل جيد حسب اختبارات Sargan و Hansen
        - في حالة المعامل المرتبط بالمتغير التابع المؤخر مساوي إلى الواحد، يمكن اللجوء إلى differenced GMM

        ### طرق التقدير الأخرى
        - هناك طرق أخرى للتقدير مثل ML و QML
        - يشترط أن لا توجد مشاكل الارتباط الذاتي وعدم تجانس التباين وغيرها
        - في حالة العينات الصغيرة، يمكن اللجوء إلى طرق تصحيح التحيز في النماذج الديناميكية مثل LSDV bias corrected

        ### الصيغة الرياضية للنموذج الديناميكي البسيط
        """)

		st.latex(r"""
        y_{it} = \alpha_i + \gamma y_{i,t-1} + \boldsymbol{x}_{it}' \boldsymbol{\beta} + \varepsilon_{it}
        """)

		st.markdown("""
        ### نماذج البانل الديناميكية في حالة N أكبر من T
        عندما يكون عدد المقاطع العرضية (N) أكبر من عدد الفترات الزمنية (T)، تظهر مشكلة التحيز في تقدير المعلمات باستخدام الطرق التقليدية. في هذه الحالة، يمكن استخدام:
        - طريقة Arellano-Bond (difference GMM)
        - طريقة Arellano-Bover/Blundell-Bond (system GMM)

        ### نماذج البانل الديناميكية في حالة T أكبر من N أو كلاهما كبيرين
        في هذه الحالة، يمكن استخدام:
        - طريقة Mean Group (MG)
        - طريقة Pooled Mean Group (PMG)
        - طريقة Dynamic Fixed Effects (DFE)

        ### الاختبارات المسبقة المهمة
        - اختبارات عدم تجانس الميول
        - اختبارات cross-sectional dependence
        - اختبارات التغير الهيكلي
        - اختبارات الاستقرارية والتغير الهيكلي
        - اختبارات التكامل المشترك (الجيل الأول والثاني والثالث)
        """)

	with col2:
		# رسم بياني لتوضيح تحيز التقدير في النماذج الديناميكية
		fig = go.Figure()

		# إنشاء بيانات وهمية
		true_gamma = 0.7
		gamma_ols = [0.9, 0.85, 0.82, 0.79, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72]
		gamma_fe = [0.55, 0.58, 0.61, 0.63, 0.65, 0.66, 0.67, 0.68, 0.69, 0.69]
		gamma_gmm = [0.72, 0.71, 0.71, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
		t_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

		# إضافة القيمة الحقيقية
		fig.add_shape(
			type='line',
			x0=0, y0=true_gamma,
			x1=55, y1=true_gamma,
			line=dict(color='#2ecc71', width=2, dash='dash')
		)

		# إضافة تقديرات مختلفة
		fig.add_trace(go.Scatter(
			x=t_values, y=gamma_ols,
			mode='lines+markers',
			name='تقدير OLS',
			line=dict(color='#e74c3c', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t_values, y=gamma_fe,
			mode='lines+markers',
			name='تقدير Fixed Effects',
			line=dict(color='#3498db', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t_values, y=gamma_gmm,
			mode='lines+markers',
			name='تقدير GMM',
			line=dict(color='#f39c12', width=2)
		))

		# إضافة تسمية للقيمة الحقيقية
		fig.add_annotation(
			x=50, y=true_gamma,
			text="القيمة الحقيقية",
			showarrow=True,
			arrowhead=1,
			ax=50, ay=-30,
			bgcolor='#2ecc71',
			bordercolor='#2ecc71',
			font=dict(color='white', size=10)
		)

		fig.update_layout(
			title="تحيز التقدير في النماذج الديناميكية حسب T",
			xaxis_title="عدد الفترات الزمنية (T)",
			yaxis_title="تقدير المعامل γ",
			height=350
		)

		st.plotly_chart(fig, use_container_width=True)

		# رسم بياني للمقارنة بين طرق التقدير المختلفة
		fig2 = go.Figure()

		methods = ['OLS', 'FE', 'Diff-GMM', 'Sys-GMM', 'LSDVC', 'MG', 'PMG', 'DFE']

		n_small_t_small = [2, 1, 4, 5, 5, 1, 3, 2]
		n_small_t_large = [2, 3, 2, 3, 4, 5, 5, 4]
		n_large_t_small = [2, 1, 5, 5, 4, 1, 3, 2]
		n_large_t_large = [3, 4, 3, 4, 4, 5, 5, 5]

		# إضافة البيانات
		fig2.add_trace(go.Bar(
			x=methods,
			y=n_small_t_small,
			name='N صغير، T صغير',
			marker_color='#3a506b'
		))

		fig2.add_trace(go.Bar(
			x=methods,
			y=n_small_t_large,
			name='N صغير، T كبير',
			marker_color='#f05454'
		))

		fig2.add_trace(go.Bar(
			x=methods,
			y=n_large_t_small,
			name='N كبير، T صغير',
			marker_color='#30475e'
		))

		fig2.add_trace(go.Bar(
			x=methods,
			y=n_large_t_large,
			name='N كبير، T كبير',
			marker_color='#7b68ee'
		))

		fig2.update_layout(
			title="مقارنة بين طرق تقدير نماذج البانل الديناميكية",
			xaxis_title="طريقة التقدير",
			yaxis_title="درجة الملاءمة (1-5)",
			height=350,
			barmode='group'
		)

		st.plotly_chart(fig2, use_container_width=True)

		# مثال على تطبيق نموذج بانل ديناميكي
		st.markdown("### مثال على تطبيق نموذج بانل ديناميكي (System GMM)")
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm

        # يتطلب تثبيت حزمة linearmodels
        from linearmodels.panel import PanelOLS, FirstDifferenceOLS, RandomEffects
        from linearmodels.panel.model import PanelGMM

        # إعداد البيانات
        df = pd.read_csv('data.csv')
        df = df.set_index(['id', 'time'])

        # تحديد المتغيرات
        endog = df['y']
        exog = sm.add_constant(df[['y_lag1', 'x1', 'x2']])

        # تعريف الأدوات
        instruments = ['y_lag2', 'y_lag3', 'x1', 'x2']

        # تقدير نموذج System GMM
        model = PanelGMM.from_formula(
            formula='y ~ 1 + y_lag1 + x1 + x2',
            data=df,
            instruments=instruments
        )

        results = model.fit()
        print(results.summary)

        # اختبار صلاحية الأدوات
        sargan_test = results.sargan
        print(f"Sargan Test: {sargan_test}")
        """
		st.code(code, language='python')

# نماذج البانل الساكنة
elif choice == "نماذج البانل الساكنة":
	st.header("نماذج البانل الساكنة (Static Panel Models)")

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        ### الهدف من النموذج
        دراسة التأثيرات الساكنة للمتغيرات المستقلة على المتغير التابع في إطار بيانات البانل، مع الاستفادة من البعدين المقطعي والزمني للبيانات.

        ### الشروط الأساسية
        - نماذج البانل الساكنة التقليدية تشترط أن تكون الميول ثابتة والثوابت متغيرة
        - يمكن استخدام نماذج البانل الساكنة في حالة N كبيرة أو T كبيرة، لكن هناك طرق حساب للتقدير تختلف حسب N وحسب T
        - في حالة العينات الصغيرة (T و N)، يمكن استخدام bias correction للنماذج الساكنة

        ### أنواع وتعديلات النموذج
        - في حالة الميول متغيرة، يمكن استخدام نماذج المعاملات المتغيرة مثل Fixed Individual Effect Variable Slopes
        - في حالة النقاط الشاذة، يمكن استخدام One-step Robust Fixed Effect
        - في حالة أحد المتغيرات المهمة هو Time-invariant Variables، يمكن استخدام Filtred Fixed Effect
        - في وجود Multicolinearity، يمكن استخدام Ridge Regression للبانل
        - في حالة مشاكل في البواقي، هناك طرق لتصحيح الانحراف المعياري مثل Driscol-Karray Methods وطرق Robust أو تغيير كامل لطرق التقدير مثل استخدام FGLS
        - في وجود Endogeneity، يمكن استخدام Fixed or Random Instrumental Variables

        ### الصيغة الرياضية للنماذج الساكنة الأساسية
        """)

		# نموذج التأثيرات الثابتة
		st.markdown("#### نموذج التأثيرات الثابتة (Fixed Effects Model)")
		st.latex(r"""
        y_{it} = \alpha_i + \boldsymbol{x}_{it}' \boldsymbol{\beta} + \varepsilon_{it}
        """)

		# نموذج التأثيرات العشوائية
		st.markdown("#### نموذج التأثيرات العشوائية (Random Effects Model)")
		st.latex(r"""
        y_{it} = \alpha + \boldsymbol{x}_{it}' \boldsymbol{\beta} + u_i + \varepsilon_{it}
        """)

		st.markdown("""
        ### الاختبارات المهمة في نماذج البانل الساكنة
        - **اختبار Hausman:** للمفاضلة بين نموذج التأثيرات الثابتة والتأثيرات العشوائية
        - **اختبار Breusch-Pagan:** للمفاضلة بين نموذج التأثيرات العشوائية ونموذج الانحدار التجميعي
        - **اختبار F:** للمفاضلة بين نموذج التأثيرات الثابتة ونموذج الانحدار التجميعي
        - **اختبارات Cross-sectional Dependence:** مثل اختبار Pesaran CD
        - **اختبارات Heteroskedasticity:** مثل اختبار Modified Wald للتأثيرات الثابتة
        - **اختبارات Serial Correlation:** مثل اختبار Wooldridge للارتباط الذاتي في بيانات البانل

        ### مشاكل النماذج الساكنة وطرق معالجتها
        - **Heteroskedasticity:** استخدام Robust Standard Errors أو FGLS
        - **Serial Correlation:** استخدام Clustered Standard Errors أو نماذج FGLS
        - **Cross-sectional Dependence:** استخدام Driscoll-Kraay Standard Errors أو Common Correlated Effects (CCE)
        - **Endogeneity:** استخدام Instrumental Variables أو نماذج GMM
        - **Outliers:** استخدام Robust Regression Methods
        """)

	with col2:
		# رسم بياني لشرح الفرق بين نماذج البانل المختلفة
		fig = go.Figure()

		# إنشاء بيانات وهمية لتوضيح الفروقات بين النماذج
		np.random.seed(42)

		# إنشاء بيانات لثلاث مجموعات
		x = np.linspace(0, 10, 20)

		# نموذج الانحدار التجميعي (نفس الميل والثابت)
		y_pooled_1 = 2 + 1.5 * x + np.random.normal(0, 1, 20)
		y_pooled_2 = 2 + 1.5 * x + np.random.normal(0, 1, 20)
		y_pooled_3 = 2 + 1.5 * x + np.random.normal(0, 1, 20)

		# نموذج التأثيرات الثابتة (نفس الميل، ثوابت مختلفة)
		y_fe_1 = 1 + 1.5 * x + np.random.normal(0, 0.7, 20)
		y_fe_2 = 3 + 1.5 * x + np.random.normal(0, 0.7, 20)
		y_fe_3 = 5 + 1.5 * x + np.random.normal(0, 0.7, 20)

		# نموذج الميول المتغيرة (ميول وثوابت مختلفة)
		y_vs_1 = 1 + 1.0 * x + np.random.normal(0, 0.5, 20)
		y_vs_2 = 3 + 1.5 * x + np.random.normal(0, 0.5, 20)
		y_vs_3 = 5 + 2.0 * x + np.random.normal(0, 0.5, 20)

		# إنشاء ثلاث رسومات بيانية منفصلة
		# 1. نموذج الانحدار التجميعي
		fig1 = go.Figure()

		fig1.add_trace(go.Scatter(
			x=x, y=y_pooled_1,
			mode='markers',
			name='المجموعة 1',
			marker=dict(color='#3a506b', size=8)
		))

		fig1.add_trace(go.Scatter(
			x=x, y=y_pooled_2,
			mode='markers',
			name='المجموعة 2',
			marker=dict(color='#f05454', size=8)
		))

		fig1.add_trace(go.Scatter(
			x=x, y=y_pooled_3,
			mode='markers',
			name='المجموعة 3',
			marker=dict(color='#30475e', size=8)
		))

		# إضافة خط الانحدار
		all_x = np.concatenate([x, x, x])
		all_y = np.concatenate([y_pooled_1, y_pooled_2, y_pooled_3])
		coef = np.polyfit(all_x, all_y, 1)
		line = coef[0] * np.linspace(0, 10, 100) + coef[1]

		fig1.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=line,
			mode='lines',
			name='خط الانحدار المجمع',
			line=dict(color='#7b68ee', width=3)
		))

		fig1.update_layout(
			title="نموذج الانحدار التجميعي",
			xaxis_title="X",
			yaxis_title="Y",
			height=200
		)

		# 2. نموذج التأثيرات الثابتة
		fig2 = go.Figure()

		fig2.add_trace(go.Scatter(
			x=x, y=y_fe_1,
			mode='markers',
			name='المجموعة 1',
			marker=dict(color='#3a506b', size=8)
		))

		fig2.add_trace(go.Scatter(
			x=x, y=y_fe_2,
			mode='markers',
			name='المجموعة 2',
			marker=dict(color='#f05454', size=8)
		))

		fig2.add_trace(go.Scatter(
			x=x, y=y_fe_3,
			mode='markers',
			name='المجموعة 3',
			marker=dict(color='#30475e', size=8)
		))

		# إضافة خطوط انحدار منفصلة بنفس الميل
		slope = 1.5

		fig2.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=slope * np.linspace(0, 10, 100) + 1,
			mode='lines',
			name='خط المجموعة 1',
			line=dict(color='#3a506b', width=3)
		))

		fig2.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=slope * np.linspace(0, 10, 100) + 3,
			mode='lines',
			name='خط المجموعة 2',
			line=dict(color='#f05454', width=3)
		))

		fig2.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=slope * np.linspace(0, 10, 100) + 5,
			mode='lines',
			name='خط المجموعة 3',
			line=dict(color='#30475e', width=3)
		))

		fig2.update_layout(
			title="نموذج التأثيرات الثابتة",
			xaxis_title="X",
			yaxis_title="Y",
			height=200,
			showlegend=False
		)

		# 3. نموذج الميول المتغيرة
		fig3 = go.Figure()

		fig3.add_trace(go.Scatter(
			x=x, y=y_vs_1,
			mode='markers',
			name='المجموعة 1',
			marker=dict(color='#3a506b', size=8)
		))

		fig3.add_trace(go.Scatter(
			x=x, y=y_vs_2,
			mode='markers',
			name='المجموعة 2',
			marker=dict(color='#f05454', size=8)
		))

		fig3.add_trace(go.Scatter(
			x=x, y=y_vs_3,
			mode='markers',
			name='المجموعة 3',
			marker=dict(color='#30475e', size=8)
		))

		# إضافة خطوط انحدار منفصلة بميول مختلفة
		fig3.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=1.0 * np.linspace(0, 10, 100) + 1,
			mode='lines',
			name='خط المجموعة 1',
			line=dict(color='#3a506b', width=3)
		))

		fig3.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=1.5 * np.linspace(0, 10, 100) + 3,
			mode='lines',
			name='خط المجموعة 2',
			line=dict(color='#f05454', width=3)
		))

		fig3.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=2.0 * np.linspace(0, 10, 100) + 5,
			mode='lines',
			name='خط المجموعة 3',
			line=dict(color='#30475e', width=3)
		))

		fig3.update_layout(
			title="نموذج الميول المتغيرة",
			xaxis_title="X",
			yaxis_title="Y",
			height=200,
			showlegend=False
		)

		# عرض الرسومات البيانية
		st.plotly_chart(fig1, use_container_width=True)
		st.plotly_chart(fig2, use_container_width=True)
		st.plotly_chart(fig3, use_container_width=True)

		# مثال على تطبيق نماذج البانل الساكنة
		st.markdown("### مثال على تطبيق نماذج البانل الساكنة")
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

        # إعداد البيانات
        df = pd.read_csv('data.csv')
        df = df.set_index(['id', 'time'])

        # 1. نموذج الانحدار التجميعي
        pooled_model = PooledOLS.from_formula('y ~ 1 + x1 + x2', data=df)
        pooled_results = pooled_model.fit()

        # 2. نموذج التأثيرات الثابتة
        fe_model = PanelOLS.from_formula('y ~ 1 + x1 + x2 + EntityEffects', data=df)
        fe_results = fe_model.fit()

        # 3. نموذج التأثيرات العشوائية
        re_model = RandomEffects.from_formula('y ~ 1 + x1 + x2', data=df)
        re_results = re_model.fit()

        # اختبار Hausman
        hausman_statistic = fe_results.test_against(re_results)

        # عرض النتائج
        print(fe_results.summary)
        print(f"Hausman Test: {hausman_statistic}")
        """
		st.code(code, language='python')

# المتناقضات في الدراسات العربية
elif choice == "المتناقضات في الدراسات العربية":
	st.header("المتناقضات في الدراسات العربية")

	st.error("""
    ### أهم المتناقضات في الدراسات العربية
    - لا يجوز استخدام اختبار جوهانسون في وجود تغيرات هيكلية
    - لا يمكن أن نجمع بين ARDL و VAR في دراسة واحدة، لأن الأول يعتمد على معادلة واحدة والآخر يعتمد على نظام من المعادلات
    - لا يمكن الجمع بين اختبار جوهانسون و Bounds Test، لأن جوهانسون يختبر العلاقة التبادلية أما Bounds Test فيختبر العلاقة في اتجاه واحد
    - لا يمكن الجمع بين اختبارات الجيل الأول والثاني في البانل للتكامل المشترك أو جذر الوحدة، لأن في وجود أو غياب Cross-sectional Dependence سيبقى لنا اختبار واحد فقط إما من الجيل الأول أو الثاني
    - لا يمكن الجمع بين ARDL-PMG و ARDL-CS، لأن وجود أو غياب Cross-sectional Dependence سيبقي لنا نموذج واحد فقط
    """)

	# إضافة رسم بياني توضيحي للتناقضات
	fig = go.Figure()

	contradictions = [
		"استخدام اختبار جوهانسون مع تغيرات هيكلية",
		"الجمع بين ARDL و VAR في نفس الدراسة",
		"الجمع بين اختبار جوهانسون و Bounds Test",
		"الجمع بين اختبارات الجيل الأول والثاني للتكامل المشترك",
		"الجمع بين ARDL-PMG و ARDL-CS"
	]

	frequency = [68, 45, 72, 53, 40]
	severity = [4, 3, 5, 4, 3]

	# تحويل الحجم إلى نطاق مناسب
	bubble_size = [s * 15 for s in severity]

	fig.add_trace(go.Scatter(
		x=frequency,
		y=[1, 2, 3, 4, 5],
		mode='markers',
		marker=dict(
			size=bubble_size,
			color=['#e74c3c', '#e67e22', '#c0392b', '#d35400', '#e74c3c'],
			opacity=0.8,
			line=dict(color='white', width=1)
		),
		text=contradictions,
		hoverinfo='text'
	))

	for i, txt in enumerate(contradictions):
		fig.add_annotation(
			x=frequency[i],
			y=i + 1,
			text=txt,
			showarrow=False,
			font=dict(size=10),
			xshift=15,
			align='left'
		)

	fig.update_layout(
		title="تكرار المتناقضات في الدراسات العربية",
		xaxis_title="تكرار الظهور في الدراسات",
		yaxis=dict(
			showticklabels=False,
			showgrid=False
		),
		height=400,
		showlegend=False
	)

	st.plotly_chart(fig, use_container_width=True)

	# نصائح لتجنب المتناقضات
	st.success("""
    ### نصائح لتجنب المتناقضات في الدراسات الاقتصادية القياسية
    1. **فهم أساسيات النموذج:** فهم الافتراضات الأساسية والشروط اللازمة لكل نموذج قبل تطبيقه.
    2. **اختيار النموذج المناسب:** اختيار النموذج الذي يتناسب مع طبيعة البيانات وأهداف الدراسة.
    3. **إجراء الاختبارات التشخيصية:** التحقق من صلاحية النموذج من خلال الاختبارات التشخيصية المناسبة.
    4. **مراعاة خصائص البيانات:** الانتباه إلى خصائص البيانات مثل الاستقرارية والتغيرات الهيكلية.
    5. **تجنب الجمع بين النماذج المتعارضة:** تجنب استخدام نماذج ذات افتراضات متعارضة في نفس الدراسة.
    """)

# ملاحظات عامة
elif choice == "ملاحظات عامة":
	st.header("ملاحظات عامة")

	st.markdown("""
    - هذا المخطط يتكلم عن أهم النماذج التفسيرية في الدراسات العربية وليست التنبؤية.
    - هذه الشروط بصفة عامة وليست مفصلة، لأن الشروط المفصلة تحتاج مخطط لكل نموذج على حدى.
    - دائماً عندنا شروط متعلقة بالبواقي وهي أن تكون خالية من المشاكل، وهذه خاصية مشتركة بين كل النماذج في المخطط.
    - المعيار الأهم لاختيار نموذج معين هو هل أهدافه تتوافق مع أهداف الدراسة وهل يستطيع أن يجيب عن إشكالية البحث، وبعدها نتكلم عن الجزئيات.
    - من شروط تعلم أي نموذج هو التركيز على الأهداف والشروط والتمرن على التطبيق ومعرفة الانتقادات والعمل على البدائل.
    """)

	# إضافة رسم بياني للعلاقة بين معايير اختيار النموذج
	fig = go.Figure()

	criteria = [
		"توافق النموذج مع أهداف الدراسة",
		"قدرة النموذج على الإجابة عن إشكالية البحث",
		"تحقق شروط تطبيق النموذج",
		"توافر البيانات اللازمة",
		"سهولة التفسير والتحليل"
	]

	importance = [5, 4.8, 4.2, 3.5, 3.2]

	fig.add_trace(go.Bar(
		x=importance,
		y=criteria,
		orientation='h',
		marker=dict(
			color=['#3a506b', '#3a506b', '#3a506b', '#3a506b', '#3a506b'],
			colorscale=[[0, '#f05454'], [1, '#3a506b']],
			line=dict(color='white', width=1)
		)
	))

	fig.update_layout(
		title="معايير اختيار النموذج القياسي المناسب (حسب الأهمية)",
		xaxis_title="درجة الأهمية",
		yaxis=dict(
			title="",
			autorange="reversed"
		),
		height=350
	)

	st.plotly_chart(fig, use_container_width=True)

	# الخاتمة والتوصيات
	st.info("""
    ### توصيات لاستخدام النماذج القياسية
    1. ضرورة فهم الأسس النظرية والافتراضات الأساسية للنماذج القياسية قبل تطبيقها.
    2. أهمية اختيار النموذج المناسب وفقاً لطبيعة البيانات وأهداف الدراسة.
    3. ضرورة إجراء الاختبارات التشخيصية للتحقق من صلاحية النموذج.
    4. تجنب استخدام النماذج المتناقضة في نفس الدراسة.
    5. الاطلاع المستمر على التطورات الحديثة في مجال النمذجة القياسية.
    """)

	st.info("إعداد: Merwan Roudane")

st.markdown("---")
st.markdown("© 2025 - النماذج القياسية الأكثر شهرة عربياً")tps://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
    }

    .main .block-container {
        direction: rtl;
        text-align: right;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        direction: rtl;
        text-align: right;
    }

    .model-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }

    .model-title {
        background-color: #3a506b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        text-align: center;
    }

    .highlight {
        background-color: #ffeaa7;
        padding: 2px 5px;
        border-radius: 3px;
    }

    .warning {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .note {
        background-color: #e0f7fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .katex-display {
        direction: ltr !important;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("<h1 style='text-align: center; color: #1e3d59;'>النماذج القياسية الأكثر شهرة عربياً</h1>",
			unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6b7b8c;'>إعداد: Merwan Roudane</h3>", unsafe_allow_html=True)

# إضافة شريط جانبي
st.sidebar.markdown("<h3 style='text-align: center;'>قائمة النماذج</h3>", unsafe_allow_html=True)
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

# إضافة معلومات إضافية في الشريط الجانبي
st.sidebar.markdown("---")
st.sidebar.markdown(
	"<div class='note'>هذا المخطط يتناول أهم النماذج التفسيرية في الدراسات العربية وليست التنبؤية</div>",
	unsafe_allow_html=True)
st.sidebar.markdown(
	"<div class='note'>الشروط المذكورة هي بصفة عامة وليست مفصلة حيث تحتاج الشروط المفصلة إلى مخطط لكل نموذج على حدى</div>",
	unsafe_allow_html=True)


# دالة لإنشاء رسم بياني للنماذج
def create_model_tree():
	fig = go.Figure()

	models = [
		"نماذج الانحدار الخطي", "نماذج الانحدار الكمي", "نماذج المعادلات الآنية",
		"نموذج VAR", "نموذج VECM", "نموذج ARDL", "نموذج NARDL",
		"نماذج البانل الديناميكية", "نماذج البانل الساكنة"
	]

	x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	y = [3, 2, 3, 1, 1, 2, 2, 3, 3]

	# إضافة النقاط
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode='markers+text',
		marker=dict(size=20, color=['#3a506b'] * len(models)),
		text=models,
		textposition="top center",
		textfont=dict(size=14, color='black', family='Arial'),
		hoverinfo='text'
	))

	# إضافة الخطوط للربط
	fig.add_shape(type="line", x0=1, y0=3, x1=3, y1=3, line=dict(color="#718096", width=2))
	fig.add_shape(type="line", x0=4, y0=1, x1=7, y1=1, line=dict(color="#718096", width=2))
	fig.add_shape(type="line", x0=8, y0=3, x1=9, y1=3, line=dict(color="#718096", width=2))

	# تنسيق الرسم البياني
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


# إنشاء رسم بياني لمقارنة شروط النماذج
def create_conditions_comparison():
	categories = ['استقرارية البيانات', 'حجم العينة', 'التوزيع الطبيعي', 'مشاكل التوصيف', 'العلاقة السببية']

	models = ['ARDL', 'VAR', 'VECM', 'نماذج البانل']
	values = [
		[3, 3, 2, 4, 5],  # ARDL
		[5, 4, 3, 3, 5],  # VAR
		[5, 4, 2, 3, 5],  # VECM
		[4, 5, 2, 4, 3],  # نماذج البانل
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


# الصفحة الرئيسية
if choice == "الرئيسية":
	st.markdown("<h2>مقدمة عن النماذج القياسية الشائعة الاستخدام عربياً</h2>", unsafe_allow_html=True)

	st.markdown("""
    <div class='note'>
    تقدم هذه الوثيقة عرضاً للنماذج القياسية الأكثر شيوعاً في الدراسات العربية مع توضيح الشروط الأساسية لاستخدامها. تشمل هذه النماذج أنواعاً مختلفة من تحليل الانحدار، ونماذج المعادلات الآنية، ونماذج السلاسل الزمنية، ونماذج البانل.
    </div>
    """, unsafe_allow_html=True)

	# عرض الرسم البياني للنماذج
	st.plotly_chart(create_model_tree(), use_container_width=True)

	# عرض مقارنة شروط النماذج
	st.plotly_chart(create_conditions_comparison(), use_container_width=True)

	# معلومات إضافية
	st.markdown("""
    <div class='model-card'>
        <h3>أهمية اختيار النموذج المناسب</h3>
        <p>يعتمد اختيار النموذج المناسب على عدة عوامل أهمها:</p>
        <ul>
            <li>هدف الدراسة (تفسيري أم تنبؤي)</li>
            <li>طبيعة البيانات (مقطعية، سلاسل زمنية، بيانات بانل)</li>
            <li>خصائص المتغيرات (استقرارية، توزيع، إلخ)</li>
            <li>العلاقة بين المتغيرات (أحادية الاتجاه، تبادلية)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# نموذج الانحدار الخطي
elif choice == "نموذج الانحدار الخطي وفروعه":
	st.markdown("<div class='model-title'><h2>نموذج الانحدار الخطي وفروعه</h2></div>", unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة الأثر المباشر للمتغيرات المستقلة على المتغير التابع.</p>

            <h3>الشروط الأساسية</h3>
            <ul>
                <li>المتغير التابع يكون continuous ويتبع التوزيع الطبيعي</li>
                <li>في النمذجة التقليدية، يكون حجم العينة أكبر من عدد المتغيرات المستقلة بكثير</li>
                <li>في النمذجة الحديثة، لا يشترط هذا الشرط</li>
                <li>غياب مشاكل التوصيف</li>
                <li>طريقة التقدير OLS تتطلب التحقق من الفرضيات الكلاسيكية</li>
            </ul>

            <h3>الصيغة الرياضية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + ... + \beta_k X_{ki} + \varepsilon_i")

		st.markdown("""
        <div class='model-card'>
            <h3>البدائل في حالات خاصة</h3>
            <ul>
                <li>في وجود التواء من جهة اليمين للمتغير التابع: استخدام Gamma regression أو Quantile regression</li>
                <li>في وجود التواء من جهة اليسار للمتغير التابع: استخدام Skewed regression أو Quantile regression</li>
                <li>في حالة وجود نقاط شاذة: استخدام Robust regression</li>
                <li>في حالة المتغير التابع عبارة عن count variable: استخدام نماذج مثل Poisson، Binomial</li>
                <li>في حالة المتغير التابع عبارة عن متغير ثنائي: استخدام نماذج مثل Logistic، Probit</li>
                <li>في حالة المتغير التابع عبارة عن فئات: استخدام Categorical regression</li>
                <li>في حالة المتغير التابع عبارة عن مجال محدد: استخدام Interval-valued regression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني للتوضيح
		fig = go.Figure()

		# إنشاء بيانات وهمية للتوضيح
		np.random.seed(42)
		x = np.linspace(0, 10, 100)
		y = 2 * x + 1 + np.random.normal(0, 2, 100)

		# إضافة نقاط البيانات
		fig.add_trace(go.Scatter(
			x=x, y=y,
			mode='markers',
			name='البيانات',
			marker=dict(color='#3a506b', size=8)
		))

		# إضافة خط الانحدار
		coef = np.polyfit(x, y, 1)
		line = coef[0] * x + coef[1]
		fig.add_trace(go.Scatter(
			x=x, y=line,
			mode='lines',
			name='خط الانحدار',
			line=dict(color='#f05454', width=3)
		))

		fig.update_layout(
			title="مثال على الانحدار الخطي البسيط",
			xaxis_title="المتغير المستقل",
			yaxis_title="المتغير التابع",
			legend_title="البيانات",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال لنموذج انحدار متعدد
		st.markdown("<h3>مثال على بنية نموذج الانحدار المتعدد</h3>", unsafe_allow_html=True)
		code = """
        import statsmodels.api as sm
        import pandas as pd

        # إعداد البيانات
        df = pd.read_csv('data.csv')

        # تحديد المتغيرات المستقلة والتابعة
        X = df[['x1', 'x2', 'x3']]
        X = sm.add_constant(X)
        y = df['y']

        # تقدير النموذج
        model = sm.OLS(y, X).fit()

        # عرض النتائج
        print(model.summary())
        """
		st.code(code, language='python')

# نموذج الانحدار الكمي
elif choice == "نموذج الانحدار الكمي":
	st.markdown("<div class='model-title'><h2>نموذج الانحدار الكمي (Quantile Regression)</h2></div>",
				unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <ul>
                <li>تقدير أثر المتغير المستقل على مختلف quantiles للمتغير التابع</li>
                <li>البحث عن الأثر غير المتماثل لتأثير المتغير المستقل على المتغير التابع عند مختلف رتب quantile</li>
            </ul>

            <h3>الشروط والخصائص</h3>
            <ul>
                <li>يستخدم في حالة وجود نقاط شاذة والتواء في المتغير التابع وحتى في المتغيرات المستقلة</li>
                <li>مناسب عند وجود اختلافات وفروقات بين قيم المتغير التابع داخل العينة (مثل متغير الأجور أو الثروة)</li>
                <li>يستخدم عند عدم التوزيع الطبيعي للبواقي في الانحدار العادي</li>
                <li>مناسب عند الرغبة في الحصول على تفسيرات لا تتعلق بالمتوسط</li>
            </ul>

            <h3>الصيغة الرياضية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"Q_{Y}(\tau|X) = \beta_0(\tau) + \beta_1(\tau) X_1 + \beta_2(\tau) X_2 + ... + \beta_k(\tau) X_k")

		st.markdown("<p>حيث τ هي رتبة الكمية (quantile) التي نهتم بها، وتتراوح من 0 إلى 1.</p>", unsafe_allow_html=True)

		st.markdown("""
        <div class='model-card'>
            <h3>تفرعات هذا النموذج</h3>
            <ul>
                <li>Quantile in Quantile Regression: نموذج أكثر مرونة يسمح بدراسة العلاقة بين الكميات للمتغيرات المستقلة والتابعة</li>
            </ul>

            <h3>ميزات استخدام الانحدار الكمي</h3>
            <ul>
                <li>أقل تأثراً بالقيم المتطرفة مقارنة بالانحدار العادي</li>
                <li>يسمح بتحليل تأثير المتغيرات المستقلة على كامل توزيع المتغير التابع وليس فقط على متوسطه</li>
                <li>لا يتطلب افتراضات قوية حول توزيع البواقي</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني للتوضيح
		np.random.seed(42)
		x = np.linspace(0, 10, 200)
		# إنشاء بيانات ذات توزيع غير متماثل
		y = 2 * x + 1 + np.random.exponential(scale=2, size=200)

		# تقدير انحدار كمي (تقريبي للعرض فقط)
		q_25 = 2 * x + 0.2  # تقريب للكمية 0.25
		q_50 = 2 * x + 1  # تقريب للكمية 0.50 (الوسيط)
		q_75 = 2 * x + 2.5  # تقريب للكمية 0.75

		fig = go.Figure()

		# إضافة نقاط البيانات
		fig.add_trace(go.Scatter(
			x=x, y=y,
			mode='markers',
			name='البيانات',
			marker=dict(color='#3a506b', size=6, opacity=0.7)
		))

		# إضافة خطوط الانحدار الكمي
		fig.add_trace(go.Scatter(
			x=x, y=q_25,
			mode='lines',
			name='الكمية 0.25',
			line=dict(color='#f05454', width=2)
		))

		fig.add_trace(go.Scatter(
			x=x, y=q_50,
			mode='lines',
			name='الكمية 0.50 (الوسيط)',
			line=dict(color='#30475e', width=2)
		))

		fig.add_trace(go.Scatter(
			x=x, y=q_75,
			mode='lines',
			name='الكمية 0.75',
			line=dict(color='#7b68ee', width=2)
		))

		fig.update_layout(
			title="مثال على الانحدار الكمي",
			xaxis_title="المتغير المستقل",
			yaxis_title="المتغير التابع",
			legend_title="البيانات والكميات",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# كود مثال
		st.markdown("<h3>مثال على تطبيق الانحدار الكمي</h3>", unsafe_allow_html=True)
		code = """
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd

        # إعداد البيانات
        df = pd.read_csv('data.csv')

        # تقدير نموذج الانحدار الكمي عند كميات مختلفة
        q_25 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.25)
        q_50 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.50)
        q_75 = smf.quantreg('y ~ x1 + x2', df).fit(q=0.75)

        # عرض النتائج
        print(q_50.summary())
        """
		st.code(code, language='python')

# نموذج المعادلات الآنية
elif choice == "نموذج المعادلات الآنية":
	st.markdown("<div class='model-title'><h2>نموذج المعادلات الآنية (Simultaneous Equations)</h2></div>",
				unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة العلاقات المتشابكة بين المتغيرات وتأثيرها الآني. حيث يمكن للمتغير أن يكون مستقلاً في معادلة وتابعاً في معادلة أخرى.</p>

            <h3>الشروط الأساسية</h3>
            <ul>
                <li>وجود خاصية Simultaneity أي المتغير المستقل في المعادلة الأولى يصبح متغير تابع في المعادلة الثانية</li>
                <li>تحقق شرط Order and Rank Conditions for Identification</li>
            </ul>

            <h3>ملاحظات مهمة</h3>
            <ul>
                <li>في حالة استخدام هذا النموذج على السلاسل الزمنية غير المستقرة وفق طرق التقدير المعروفة، فإن Estimators تفقد الكفاءة (efficiency)</li>
                <li>في حالة متغيرات غير مستقرة ومتكاملة، نستخدم منهجية Hisao 1997</li>
            </ul>

            <h3>الصيغة الرياضية لنظام المعادلات الآنية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        \begin{align}
        Y_1 &= \beta_{10} + \beta_{12}Y_2 + \gamma_{11}X_1 + \gamma_{12}X_2 + \varepsilon_1 \\
        Y_2 &= \beta_{20} + \beta_{21}Y_1 + \gamma_{21}X_1 + \gamma_{22}X_2 + \varepsilon_2
        \end{align}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>طرق التقدير</h3>
            <ul>
                <li>Two-Stage Least Squares (2SLS)</li>
                <li>Three-Stage Least Squares (3SLS)</li>
                <li>Limited Information Maximum Likelihood (LIML)</li>
                <li>Full Information Maximum Likelihood (FIML)</li>
                <li>Generalized Method of Moments (GMM)</li>
            </ul>

            <h3>مثال على نظام معادلات آنية</h3>
            <p>نموذج العرض والطلب في الاقتصاد:</p>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        \begin{align}
        Q^d &= \alpha_0 + \alpha_1 P + \alpha_2 Y + \varepsilon_1 \quad \text{(معادلة الطلب)} \\
        Q^s &= \beta_0 + \beta_1 P + \beta_2 W + \varepsilon_2 \quad \text{(معادلة العرض)} \\
        Q^d &= Q^s \quad \text{(شرط التوازن)}
        \end{align}
        """)

		st.markdown("""
        <div class='note'>
        حيث:
        <ul>
            <li>Q^d: الكمية المطلوبة</li>
            <li>Q^s: الكمية المعروضة</li>
            <li>P: السعر (متغير داخلي)</li>
            <li>Y: الدخل (متغير خارجي يؤثر على الطلب)</li>
            <li>W: تكلفة الإنتاج (متغير خارجي يؤثر على العرض)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني توضيحي للعلاقات المتشابكة
		nodes = ['Y₁', 'Y₂', 'X₁', 'X₂']
		edges = [('Y₁', 'Y₂'), ('Y₂', 'Y₁'), ('X₁', 'Y₁'), ('X₁', 'Y₂'), ('X₂', 'Y₁'), ('X₂', 'Y₂')]

		# إنشاء الرسم البياني التوضيحي
		G = {node: [] for node in nodes}
		for edge in edges:
			G[edge[0]].append(edge[1])

		# تحديد مواقع النقاط
		pos = {
			'Y₁': [0, 0.5],
			'Y₂': [1, 0.5],
			'X₁': [0.25, 1],
			'X₂': [0.75, 1]
		}

		fig = go.Figure()

		# إضافة الحواف
		for source, targets in G.items():
			for target in targets:
				fig.add_trace(go.Scatter(
					x=[pos[source][0], pos[target][0]],
					y=[pos[source][1], pos[target][1]],
					mode='lines',
					line=dict(width=2, color='#718096'),
					hoverinfo='none'
				))

		# إضافة النقاط
		node_x = [pos[node][0] for node in nodes]
		node_y = [pos[node][1] for node in nodes]

		colors = ['#f05454', '#f05454', '#30475e', '#30475e']

		fig.add_trace(go.Scatter(
			x=node_x,
			y=node_y,
			mode='markers+text',
			marker=dict(
				size=30,
				color=colors,
				line=dict(width=2, color='white')
			),
			text=nodes,
			textposition="middle center",
			textfont=dict(size=20, color='white'),
			hoverinfo='text',
			hovertext=[
				"المتغير التابع في المعادلة الأولى",
				"المتغير التابع في المعادلة الثانية",
				"متغير مستقل خارجي",
				"متغير مستقل خارجي"
			]
		))

		fig.update_layout(
			title="العلاقات المتشابكة في نموذج المعادلات الآنية",
			showlegend=False,
			height=400,
			plot_bgcolor='#f9f9f9',
			xaxis=dict(
				showticklabels=False,
				showgrid=False,
				zeroline=False,
				range=[-0.1, 1.1]
			),
			yaxis=dict(
				showticklabels=False,
				showgrid=False,
				zeroline=False,
				range=[0.4, 1.1]
			)
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج المعادلات الآنية
		st.markdown("<h3>مثال على تطبيق نموذج المعادلات الآنية</h3>", unsafe_allow_html=True)
		code = """
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.gmm import IV2SLS
        import pandas as pd

        # إعداد البيانات
        df = pd.read_csv('data.csv')

        # تعريف المتغيرات
        endog = df['y1']            # المتغير التابع في المعادلة الأولى
        exog = df[['const', 'y2']]  # المتغيرات المستقلة (بما فيها المتغير الداخلي)
        instruments = df[['const', 'x1', 'x2']]  # الأدوات (بما فيها المتغيرات الخارجية)

        # تقدير النموذج باستخدام طريقة 2SLS
        model = IV2SLS(endog, exog, instruments).fit()

        # عرض النتائج
        print(model.summary())
        """
		st.code(code, language='python')

# نموذج VAR
elif choice == "نموذج VAR":
	st.markdown("<div class='model-title'><h2>نموذج VAR (Vector Autoregression)</h2></div>", unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة العلاقة التبادلية بين المتغيرات في إطار السلاسل الزمنية، حيث يكون الهدف الأساسي هو التنبؤ بالإضافة إلى تحليل الصدمات. في هذا النموذج، تعتبر كل المتغيرات تابعة.</p>

            <h3>الشروط المتعلقة بالاستقرارية</h3>
            <ul>
                <li>المنهجية التقليدية:
                    <ul>
                        <li>كل المتغيرات مستقرة في الفرق الأول أو الفرق الثاني وعدم وجود تكامل مشترك (أو عدم صلاحية نموذج VECM)</li>
                        <li>كل المتغيرات مستقرة في المستوى في إطار نظام من المعادلات</li>
                    </ul>
                </li>
                <li>المنهجية الحديثة:
                    <ul>
                        <li>تطورات للنموذج حيث لا يشترط أصلاً دراسة الاستقرارية في إطار VAR-Integrated أو VAR-TVP</li>
                    </ul>
                </li>
            </ul>

            <h3>أنواع وتعديلات النموذج</h3>
            <ul>
                <li>في وجود متغيرات مستقلة، ننتقل من VAR إلى VARx</li>
                <li>إذا كان الهدف تحليل الصدمات، يمكن استخدام SVAR (Structural VAR)</li>
            </ul>

            <h3>الصيغة الرياضية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        \begin{pmatrix} y_{1t} \\ y_{2t} \\ \vdots \\ y_{nt} \end{pmatrix} = 
        \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix} +
        \begin{pmatrix} 
        \phi_{11}^1 & \phi_{12}^1 & \cdots & \phi_{1n}^1 \\
        \phi_{21}^1 & \phi_{22}^1 & \cdots & \phi_{2n}^1 \\
        \vdots & \vdots & \ddots & \vdots \\
        \phi_{n1}^1 & \phi_{n2}^1 & \cdots & \phi_{nn}^1
        \end{pmatrix}
        \begin{pmatrix} y_{1,t-1} \\ y_{2,t-1} \\ \vdots \\ y_{n,t-1} \end{pmatrix} + \cdots +
        \begin{pmatrix} 
        \phi_{11}^p & \phi_{12}^p & \cdots & \phi_{1n}^p \\
        \phi_{21}^p & \phi_{22}^p & \cdots & \phi_{2n}^p \\
        \vdots & \vdots & \ddots & \vdots \\
        \phi_{n1}^p & \phi_{n2}^p & \cdots & \phi_{nn}^p
        \end{pmatrix}
        \begin{pmatrix} y_{1,t-p} \\ y_{2,t-p} \\ \vdots \\ y_{n,t-p} \end{pmatrix} +
        \begin{pmatrix} \varepsilon_{1t} \\ \varepsilon_{2t} \\ \vdots \\ \varepsilon_{nt} \end{pmatrix}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>استخدامات النموذج</h3>
            <ul>
                <li>التنبؤ بالقيم المستقبلية للمتغيرات</li>
                <li>تحليل الصدمات وتأثيرها على المتغيرات</li>
                <li>تحليل تفكيك التباين (Variance Decomposition)</li>
                <li>تحليل دوال الاستجابة النبضية (Impulse Response Functions)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني لدالة الاستجابة النبضية (IRF)
		fig = go.Figure()

		# إنشاء بيانات وهمية لدالة الاستجابة النبضية
		periods = list(range(11))
		irf_values = [0, 0.05, 0.1, 0.14, 0.16, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01]
		confidence_upper = [v + 0.05 for v in irf_values]
		confidence_lower = [max(0, v - 0.05) for v in irf_values]

		# إضافة منطقة فاصل الثقة
		fig.add_trace(go.Scatter(
			x=periods + periods[::-1],
			y=confidence_upper + confidence_lower[::-1],
			fill='toself',
			fillcolor='rgba(58, 80, 107, 0.2)',
			line=dict(color='rgba(255, 255, 255, 0)'),
			hoverinfo='skip',
			showlegend=False
		))

		# إضافة دالة الاستجابة النبضية
		fig.add_trace(go.Scatter(
			x=periods,
			y=irf_values,
			mode='lines+markers',
			line=dict(color='#3a506b', width=3),
			marker=dict(size=8),
			name='دالة الاستجابة النبضية'
		))

		# إضافة خط الصفر
		fig.add_shape(
			type='line',
			x0=0, y0=0,
			x1=10, y1=0,
			line=dict(color='#718096', width=1, dash='dash')
		)

		fig.update_layout(
			title="مثال على دالة الاستجابة النبضية (IRF)",
			xaxis_title="الفترات الزمنية",
			yaxis_title="استجابة المتغير",
			height=300
		)

		st.plotly_chart(fig, use_container_width=True)

		# رسم بياني لتفكيك التباين
		fig = go.Figure()

		# إنشاء بيانات وهمية لتفكيك التباين
		periods = list(range(1, 11))
		var1 = [100, 90, 80, 75, 70, 68, 65, 63, 60, 58]
		var2 = [0, 5, 10, 12, 15, 16, 18, 19, 21, 22]
		var3 = [0, 5, 10, 13, 15, 16, 17, 18, 19, 20]

		# إضافة المساهمات المختلفة
		fig.add_trace(go.Bar(
			x=periods,
			y=var1,
			name='المتغير 1',
			marker_color='#3a506b'
		))

		fig.add_trace(go.Bar(
			x=periods,
			y=var2,
			name='المتغير 2',
			marker_color='#f05454'
		))

		fig.add_trace(go.Bar(
			x=periods,
			y=var3,
			name='المتغير 3',
			marker_color='#30475e'
		))

		fig.update_layout(
			title="مثال على تفكيك التباين",
			xaxis_title="الفترات الزمنية",
			yaxis_title="نسبة المساهمة (%)",
			barmode='stack',
			height=300
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج VAR
		st.markdown("<h3>مثال على تطبيق نموذج VAR</h3>", unsafe_allow_html=True)
		code = """
        import pandas as pd
        from statsmodels.tsa.api import VAR

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # تحديد عدد الفجوات الزمنية المثلى
        model = VAR(df)
        results = model.select_order(maxlags=10)

        # تقدير النموذج
        var_model = model.fit(results.aic)

        # التنبؤ
        forecast = var_model.forecast(df.values[-results.aic:], steps=5)

        # تحليل دوال الاستجابة النبضية
        irf = var_model.irf(10)
        irf.plot()

        # تحليل تفكيك التباين
        fevd = var_model.fevd(10)
        fevd.plot()
        """
		st.code(code, language='python')

# نموذج VECM
elif choice == "نموذج VECM":
	st.markdown("<div class='model-title'><h2>نموذج VECM (Vector Error Correction Model)</h2></div>",
				unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة العلاقة التبادلية بين المتغيرات المعتمدة على الأجلين القصير والطويل في إطار نظام من المعادلات.</p>

            <h3>الشروط الأساسية</h3>
            <ul>
                <li>يجب أن تكون كل المتغيرات مستقرة في الفرق الأول أو كلها في الفرق الثاني</li>
                <li>يجب أن تتحقق شروط identification</li>
                <li>يجب تحقق شروط متعلقة بـ exogeneity of variables</li>
                <li>يجب أن يكون معامل تصحيح الخطأ سالب ومعنوي</li>
            </ul>

            <h3>أنواع وتعديلات النموذج</h3>
            <ul>
                <li>في حالة وجود متغيرات مستقلة، يصبح نموذج VECM بـ VECMX</li>
                <li>إذا كان هدف الدراسة هو تحليل الصدمات، يمكن الانتقال إلى SVECM</li>
            </ul>

            <h3>الصيغة الرياضية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        \Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \varepsilon_t
        """)

		st.markdown("<p>حيث:</p>", unsafe_allow_html=True)
		st.latex(r"""
        \begin{align}
        \alpha &: \text{مصفوفة معاملات التعديل (معاملات تصحيح الخطأ)} \\
        \beta &: \text{مصفوفة متجهات التكامل المشترك} \\
        \Gamma_i &: \text{مصفوفة معاملات الآثار قصيرة الأجل}
        \end{align}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>العلاقة بين VAR و VECM</h3>
            <p>يمكن اعتبار VECM حالة خاصة من نموذج VAR مع قيود على المعاملات طويلة الأجل. وتحديداً، VECM هو نموذج VAR مقيد بوجود علاقة تكامل مشترك بين المتغيرات.</p>

            <h3>مراحل تطبيق نموذج VECM</h3>
            <ol>
                <li>اختبار استقرارية السلاسل الزمنية والتأكد من أنها متكاملة من الدرجة الأولى I(1)</li>
                <li>تحديد العدد الأمثل للفجوات الزمنية باستخدام معايير المعلومات</li>
                <li>اختبار وجود تكامل مشترك باستخدام منهجية جوهانسن</li>
                <li>تقدير نموذج VECM</li>
                <li>اختبار صلاحية النموذج من خلال فحص البواقي ومعامل تصحيح الخطأ</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني توضيحي لآلية عمل VECM
		np.random.seed(42)
		t = np.linspace(0, 10, 200)

		# إنشاء سلسلتين زمنيتين متكاملتين مشتركاً
		equilibrium = 2 * t
		y1 = equilibrium + np.random.normal(0, 1, 200)
		y2 = equilibrium + np.random.normal(0, 1, 200)

		# إضافة انحراف في نقطة معينة ثم تصحيح
		shock_point = 100
		y1[shock_point:shock_point + 30] += np.linspace(0, 5, 30)
		y1[shock_point + 30:] += 5 - 5 * np.exp(-0.1 * np.arange(70))

		fig = go.Figure()

		# إضافة السلاسل الزمنية
		fig.add_trace(go.Scatter(
			x=t, y=y1,
			mode='lines',
			name='السلسلة الزمنية 1',
			line=dict(color='#3a506b', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t, y=y2,
			mode='lines',
			name='السلسلة الزمنية 2',
			line=dict(color='#f05454', width=2)
		))

		# إضافة التوازن طويل الأجل
		fig.add_trace(go.Scatter(
			x=t, y=equilibrium,
			mode='lines',
			name='التوازن طويل الأجل',
			line=dict(color='#30475e', width=2, dash='dash')
		))

		# إشارة إلى نقطة الصدمة
		fig.add_annotation(
			x=t[shock_point], y=y1[shock_point],
			text="الصدمة",
			showarrow=True,
			arrowhead=1,
			ax=0, ay=-40
		)

		# إشارة إلى عملية التصحيح
		fig.add_annotation(
			x=t[shock_point + 50], y=y1[shock_point + 50],
			text="تصحيح الخطأ",
			showarrow=True,
			arrowhead=1,
			ax=0, ay=-40
		)

		fig.update_layout(
			title="آلية عمل نموذج تصحيح الخطأ (VECM)",
			xaxis_title="الزمن",
			yaxis_title="القيمة",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج VECM
		st.markdown("<h3>مثال على تطبيق نموذج VECM</h3>", unsafe_allow_html=True)
		code = """
        import pandas as pd
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.vector_ar.vecm import VECM
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # اختبار التكامل المشترك
        johansen_test = coint_johansen(df, 0, 2)

        # تحديد عدد علاقات التكامل المشترك
        trace_stat = johansen_test.lr1
        trace_crit = johansen_test.cvt
        r = sum(trace_stat > trace_crit[:, 1])

        # تقدير نموذج VECM
        model = VECM(df, k_ar_diff=2, coint_rank=r, deterministic='ci')
        results = model.fit()

        # عرض النتائج
        print(results.summary())

        # استخراج معاملات تصحيح الخطأ
        alpha = results.alpha
        print("معاملات تصحيح الخطأ:")
        print(alpha)
        """
		st.code(code, language='python')

# نموذج ARDL
elif choice == "نموذج ARDL":
	st.markdown("<div class='model-title'><h2>نموذج ARDL (Autoregressive Distributed Lag)</h2></div>",
				unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة التأثير الديناميكي والعلاقة طويلة الأجل مع تقدير قصيرة الأجل في إطار معادلة واحدة (لا يوجد feedback بين المتغير التابع والمتغيرات المستقلة).</p>

            <h3>الشروط الأساسية</h3>
            <ul>
                <li>الاستقرارية في المستوى أو الفرق الأول على الأكثر (لا توجد متغيرات مستقرة في الفرق الثاني)</li>
                <li>حجم العينة على الأقل 30</li>
                <li>في حالة حجم العينة أقل من 30، نستخدم ARDL BOOTSTRAPPING</li>
            </ul>

            <h3>أنواع وتعديلات النموذج</h3>
            <ul>
                <li>في حالة المتغير التابع مستقر في المستوى، نستخدم AUGMENTED ARDL</li>
                <li>في وجود عدة تغيرات هيكلية، نستخدم FOURRIER ARDL أو استخدام DUMMIES</li>
                <li>في حالة عدم وجود علاقة طويلة الأجل، يمكن استخدام DIFFERENCED ARDL كبديل</li>
            </ul>

            <h3>الصيغة الرياضية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        \begin{align}
        \Delta y_t &= \alpha_0 + \alpha_1 t + \delta_1 y_{t-1} + \delta_2 x_{t-1} + \delta_3 z_{t-1} + ... \\
        &+ \sum_{i=1}^{p} \beta_i \Delta y_{t-i} + \sum_{i=0}^{q} \gamma_i \Delta x_{t-i} + \sum_{i=0}^{r} \theta_i \Delta z_{t-i} + ... + \varepsilon_t
        \end{align}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>مزايا نموذج ARDL</h3>
            <ul>
                <li>يمكن استخدامه مع متغيرات ذات درجات تكامل مختلفة (I(0) و I(1) ولكن ليس I(2))</li>
                <li>يسمح بتقدير العلاقات طويلة وقصيرة الأجل في معادلة واحدة</li>
                <li>يعالج مشكلة Endogeneity وارتباط البواقي من خلال إدراج عدد كافٍ من الفجوات الزمنية</li>
                <li>يمكن استخدامه مع عينات صغيرة نسبياً</li>
            </ul>

            <h3>اختبارات الحدود (Bounds Test)</h3>
            <p>يستخدم اختبار الحدود ARDL Bounds Test للتحقق من وجود علاقة توازن طويلة الأجل بين المتغيرات، بغض النظر عن كونها I(0) أو I(1).</p>

            <p>الفرضية الصفرية: لا توجد علاقة تكامل (توازن) طويلة الأجل.</p>
            <p>الفرضية البديلة: توجد علاقة تكامل طويلة الأجل.</p>

            <h3>مراحل تطبيق نموذج ARDL</h3>
            <ol>
                <li>التأكد من استقرارية المتغيرات (I(0) أو I(1) وليس I(2))</li>
                <li>تحديد العدد الأمثل للفجوات الزمنية باستخدام معايير المعلومات</li>
                <li>تقدير نموذج ARDL</li>
                <li>إجراء اختبار الحدود Bounds Test للتحقق من وجود علاقة توازن طويلة الأجل</li>
                <li>تقدير العلاقة طويلة الأجل ونموذج تصحيح الخطأ</li>
                <li>إجراء اختبارات التشخيص للتحقق من صلاحية النموذج</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني توضيحي لاختبار الحدود
		fig = go.Figure()

		# إنشاء بيانات وهمية
		f_stat = 5.2
		lower_bound_1 = 2.8
		upper_bound_1 = 3.8
		lower_bound_5 = 2.1
		upper_bound_5 = 3.0
		lower_bound_10 = 1.8
		upper_bound_10 = 2.7

		# إضافة القيمة المحسوبة لإحصائية F
		fig.add_trace(go.Scatter(
			x=['القيمة المحسوبة'],
			y=[f_stat],
			mode='markers',
			marker=dict(size=15, color='#f05454'),
			name='إحصائية F المحسوبة'
		))

		# إضافة حدود الاختبار
		fig.add_trace(go.Scatter(
			x=['1%', '5%', '10%'],
			y=[lower_bound_1, lower_bound_5, lower_bound_10],
			mode='lines+markers',
			marker=dict(size=10, color='#3a506b'),
			line=dict(width=2, color='#3a506b'),
			name='الحد الأدنى'
		))

		fig.add_trace(go.Scatter(
			x=['1%', '5%', '10%'],
			y=[upper_bound_1, upper_bound_5, upper_bound_10],
			mode='lines+markers',
			marker=dict(size=10, color='#30475e'),
			line=dict(width=2, color='#30475e'),
			name='الحد الأعلى'
		))

		# تحديد المناطق
		fig.add_shape(
			type='rect',
			x0=-0.5, y0=0,
			x1=3.5, y1=lower_bound_1,
			fillcolor='rgba(255, 0, 0, 0.1)',
			line=dict(width=0),
			layer='below'
		)

		fig.add_shape(
			type='rect',
			x0=-0.5, y0=upper_bound_1,
			x1=3.5, y1=7,
			fillcolor='rgba(0, 255, 0, 0.1)',
			line=dict(width=0),
			layer='below'
		)

		fig.add_shape(
			type='rect',
			x0=-0.5, y0=lower_bound_1,
			x1=3.5, y1=upper_bound_1,
			fillcolor='rgba(255, 255, 0, 0.1)',
			line=dict(width=0),
			layer='below'
		)

		fig.update_layout(
			title="مثال على اختبار الحدود (Bounds Test)",
			xaxis_title="مستويات المعنوية",
			yaxis_title="قيمة إحصائية F",
			height=300,
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=1.02,
				xanchor="right",
				x=1
			)
		)

		# إضافة تفسير المناطق
		fig.add_annotation(
			x=2.5, y=6.5,
			text="منطقة رفض الفرضية الصفرية<br>(وجود علاقة تكامل مشترك)",
			showarrow=False,
			bgcolor='rgba(0, 255, 0, 0.1)',
			bordercolor='rgba(0, 255, 0, 0.5)',
			borderwidth=1,
			borderpad=4,
			font=dict(size=10)
		)

		fig.add_annotation(
			x=2.5, y=1,
			text="منطقة قبول الفرضية الصفرية<br>(عدم وجود علاقة تكامل مشترك)",
			showarrow=False,
			bgcolor='rgba(255, 0, 0, 0.1)',
			bordercolor='rgba(255, 0, 0, 0.5)',
			borderwidth=1,
			borderpad=4,
			font=dict(size=10)
		)

		fig.add_annotation(
			x=2.5, y=3.3,
			text="منطقة غير حاسمة",
			showarrow=False,
			bgcolor='rgba(255, 255, 0, 0.1)',
			bordercolor='rgba(255, 255, 0, 0.5)',
			borderwidth=1,
			borderpad=4,
			font=dict(size=10)
		)

		st.plotly_chart(fig, use_container_width=True)

		# مثال على تطبيق نموذج ARDL
		st.markdown("<h3>مثال على تطبيق نموذج ARDL</h3>", unsafe_allow_html=True)
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.tsa.ardl import ardl_select_order, ARDL

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # تحديد العدد الأمثل للفجوات الزمنية
        order_select = ardl_select_order(
            endog=df['y'],
            exog=df[['x1', 'x2']],
            maxlag=4,
            maxorder=4,
            trend='c',
            ic='aic'
        )

        # تقدير نموذج ARDL
        ardl_model = ARDL(
            endog=df['y'],
            exog=df[['x1', 'x2']],
            lags=order_select.lags,
            order=order_select.order,
            trend='c'
        )

        ardl_results = ardl_model.fit()
        print(ardl_results.summary())

        # إجراء اختبار الحدود
        bounds_test = ardl_results.bounds_test()
        print(bounds_test)

        # استخراج العلاقة طويلة الأجل
        long_run = ardl_results.long_run()
        print(long_run)
        """
		st.code(code, language='python')

# نموذج NARDL
elif choice == "نموذج NARDL":
	st.markdown("<div class='model-title'><h2>نموذج NARDL (Nonlinear ARDL)</h2></div>", unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة التأثيرات الديناميكية غير المتماثلة للمتغيرات المستقلة على المتغير التابع في الأجل الطويل والقصير.</p>

            <h3>الشروط الأساسية</h3>
            <ul>
                <li>نفس الشروط المتعلقة بنموذج ARDL فيما يتعلق بالاستقرارية (I(0) أو I(1) وليس I(2))</li>
                <li>يمكن أن يكون هناك feedback بين المتغيرات المستقلة والمتغير التابع</li>
            </ul>

            <h3>حالات خاصة وتعديلات</h3>
            <ul>
                <li>في وجود مشكل singularity، يمكن الانتقال من طريقة التقدير بالخطوة الواحدة إلى طريقة التقدير بالخطوتين (two-step)</li>
                <li>في حالة سيطرة تأثيرات موجبة على التأثيرات السالبة أو العكس، يمكن اللجوء إلى نماذج Multiple or Threshold ARDL</li>
                <li>هناك نماذج أخرى غير شائعة في الأبحاث مثل Fuzzy ARDL أو Wavelet ARDL</li>
            </ul>

            <h3>الصيغة الرياضية</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        \begin{align}
        \Delta y_t &= \alpha_0 + \alpha_1 t + \delta_1 y_{t-1} + \delta_2^+ x^+_{t-1} + \delta_2^- x^-_{t-1} + \ldots \\
        &+ \sum_{i=1}^{p} \beta_i \Delta y_{t-i} + \sum_{i=0}^{q} (\gamma_i^+ \Delta x^+_{t-i} + \gamma_i^- \Delta x^-_{t-i}) + \ldots + \varepsilon_t
        \end{align}
        """)

		st.markdown("<p>حيث:</p>", unsafe_allow_html=True)
		st.latex(r"""
        \begin{align}
        x_t^+ &= \sum_{j=1}^{t} \Delta x_j^+ = \sum_{j=1}^{t} \max(\Delta x_j, 0) \\
        x_t^- &= \sum_{j=1}^{t} \Delta x_j^- = \sum_{j=1}^{t} \min(\Delta x_j, 0)
        \end{align}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>الفرق بين ARDL و NARDL</h3>
            <p>الفرق الرئيسي بين ARDL و NARDL هو أن NARDL يسمح بتأثيرات غير متماثلة للزيادات والانخفاضات في المتغيرات المستقلة. يتم تحقيق ذلك من خلال تفكيك المتغيرات المستقلة إلى مكونات موجبة وسالبة.</p>

            <h3>اختبار عدم التماثل</h3>
            <p>بعد تقدير نموذج NARDL، يمكن اختبار وجود تأثيرات غير متماثلة طويلة الأجل من خلال اختبار الفرضية:</p>
            <p>الفرضية الصفرية (تماثل طويل الأجل): $\frac{\delta_2^+}{-\delta_1} = \frac{\delta_2^-}{-\delta_1}$</p>
            <p>وبالمثل، يمكن اختبار عدم التماثل قصير الأجل من خلال اختبار الفرضية:</p>
            <p>الفرضية الصفرية (تماثل قصير الأجل): $\sum_{i=0}^{q} \gamma_i^+ = \sum_{i=0}^{q} \gamma_i^-$</p>

            <h3>مراحل تطبيق نموذج NARDL</h3>
            <ol>
                <li>التأكد من استقرارية المتغيرات (I(0) أو I(1) وليس I(2))</li>
                <li>تفكيك المتغيرات المستقلة إلى مكونات موجبة وسالبة</li>
                <li>تحديد العدد الأمثل للفجوات الزمنية</li>
                <li>تقدير نموذج NARDL</li>
                <li>إجراء اختبار الحدود للتحقق من وجود علاقة توازن طويلة الأجل</li>
                <li>اختبار عدم التماثل في الأجلين الطويل والقصير</li>
                <li>تحليل المعاملات وتفسير النتائج</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني توضيحي للتأثيرات غير المتماثلة
		np.random.seed(42)
		t = np.linspace(0, 10, 100)
		x = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, 100)

		# تفكيك المتغير إلى مكونات موجبة وسالبة
		dx = np.diff(x, prepend=x[0])
		dx_pos = np.maximum(dx, 0)
		dx_neg = np.minimum(dx, 0)

		x_pos = np.cumsum(dx_pos)
		x_neg = np.cumsum(dx_neg)

		# تأثيرات مختلفة للتغيرات الموجبة والسالبة
		y_pos_effect = 0.8 * x_pos
		y_neg_effect = 1.5 * x_neg

		# المتغير التابع النهائي
		y = y_pos_effect + y_neg_effect + np.random.normal(0, 0.2, 100)

		fig = go.Figure()

		# إضافة المتغير المستقل
		fig.add_trace(go.Scatter(
			x=t, y=x,
			mode='lines',
			name='المتغير المستقل (x)',
			line=dict(color='#3a506b', width=2)
		))

		# إضافة المكونات الموجبة والسالبة
		fig.add_trace(go.Scatter(
			x=t, y=x_pos,
			mode='lines',
			name='المكون الموجب (x⁺)',
			line=dict(color='#2ecc71', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t, y=x_neg,
			mode='lines',
			name='المكون السالب (x⁻)',
			line=dict(color='#e74c3c', width=2)
		))

		# إضافة المتغير التابع
		fig.add_trace(go.Scatter(
			x=t, y=y,
			mode='lines',
			name='المتغير التابع (y)',
			line=dict(color='#f05454', width=2)
		))

		fig.update_layout(
			title="تفكيك المتغير المستقل في نموذج NARDL",
			xaxis_title="الزمن",
			yaxis_title="القيمة",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

		# رسم بياني لتوضيح التأثيرات التراكمية غير المتماثلة
		t_sim = np.arange(20)

		# افتراض وجود صدمة إيجابية وصدمة سلبية
		shock_pos = np.zeros(20)
		shock_pos[5] = 1  # صدمة إيجابية في الفترة 5

		shock_neg = np.zeros(20)
		shock_neg[12] = -1  # صدمة سلبية في الفترة 12

		# التأثيرات التراكمية المختلفة
		cum_effect_pos = np.zeros(20)
		cum_effect_neg = np.zeros(20)

		for i in range(5, 20):
			if i == 5:
				cum_effect_pos[i] = 0.3
			elif i > 5 and i < 10:
				cum_effect_pos[i] = cum_effect_pos[i - 1] + 0.15 * (1 - cum_effect_pos[i - 1])
			else:
				cum_effect_pos[i] = cum_effect_pos[i - 1] + 0.05 * (0.8 - cum_effect_pos[i - 1])

		for i in range(12, 20):
			if i == 12:
				cum_effect_neg[i] = -0.5
			elif i > 12 and i < 15:
				cum_effect_neg[i] = cum_effect_neg[i - 1] - 0.2 * (-1.2 - cum_effect_neg[i - 1])
			else:
				cum_effect_neg[i] = cum_effect_neg[i - 1] - 0.1 * (-1.5 - cum_effect_neg[i - 1])

		fig2 = go.Figure()

		# إضافة الصدمات
		fig2.add_trace(go.Scatter(
			x=t_sim, y=shock_pos,
			mode='lines+markers',
			name='صدمة إيجابية',
			line=dict(color='#2ecc71', width=2)
		))

		fig2.add_trace(go.Scatter(
			x=t_sim, y=shock_neg,
			mode='lines+markers',
			name='صدمة سلبية',
			line=dict(color='#e74c3c', width=2)
		))

		# إضافة التأثيرات التراكمية
		fig2.add_trace(go.Scatter(
			x=t_sim, y=cum_effect_pos,
			mode='lines',
			name='التأثير التراكمي للصدمة الإيجابية',
			line=dict(color='#2ecc71', width=2, dash='dash')
		))

		fig2.add_trace(go.Scatter(
			x=t_sim, y=cum_effect_neg,
			mode='lines',
			name='التأثير التراكمي للصدمة السلبية',
			line=dict(color='#e74c3c', width=2, dash='dash')
		))

		fig2.update_layout(
			title="التأثيرات التراكمية غير المتماثلة للصدمات",
			xaxis_title="الفترات الزمنية",
			yaxis_title="التأثير",
			height=300
		)

		st.plotly_chart(fig2, use_container_width=True)

		# مثال على تطبيق نموذج NARDL
		st.markdown("<h3>مثال على تطبيق نموذج NARDL</h3>", unsafe_allow_html=True)
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm

        # إعداد البيانات
        df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

        # تفكيك المتغير المستقل إلى مكونات موجبة وسالبة
        df['dx'] = df['x'].diff().fillna(0)
        df['dx_pos'] = df['dx'].apply(lambda x: max(x, 0))
        df['dx_neg'] = df['dx'].apply(lambda x: min(x, 0))

        df['x_pos'] = df['dx_pos'].cumsum()
        df['x_neg'] = df['dx_neg'].cumsum()

        # تقدير نموذج NARDL
        y = df['y']
        X = sm.add_constant(df[['y_lag1', 'x_pos_lag1', 'x_neg_lag1', 
                                'dy_lag1', 'dx_pos', 'dx_pos_lag1', 
                                'dx_neg', 'dx_neg_lag1']])

        model = sm.OLS(y, X).fit()
        print(model.summary())

        # اختبار التكامل المشترك (اختبار الحدود)
        # ...

        # اختبار عدم التماثل طويل الأجل
        # ...
        """
		st.code(code, language='python')

# نماذج البانل الديناميكية
elif choice == "نماذج البانل الديناميكية":
	st.markdown("<div class='model-title'><h2>نماذج البانل الديناميكية (Dynamic Panel Models)</h2></div>",
				unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>فهم وتقدير العلاقة الديناميكية بين المتغيرات لفهم سلوك المتغيرات عبر الزمن، سواء في إطار معادلة واحدة أو نظام من المعادلات.</p>

            <h3>الشروط الأساسية للتقدير بطريقة GMM</h3>
            <ul>
                <li>يفترض أن المعامل المرتبط بالمتغير التابع يجب أن يكون أصغر من 1</li>
                <li>من المستحسن أن لا يكون هناك cross-sectional dependence</li>
                <li>يجب أن تكون شروط العزوم معرفة (شرط نظري)</li>
                <li>يجب أن تكون instruments ليس كثيرة جداً وتكون معرفة ومحددة بشكل جيد حسب اختبارات Sargan و Hansen</li>
                <li>في حالة المعامل المرتبط بالمتغير التابع المؤخر مساوي إلى الواحد، يمكن اللجوء إلى differenced GMM</li>
            </ul>

            <h3>طرق التقدير الأخرى</h3>
            <ul>
                <li>هناك طرق أخرى للتقدير مثل ML و QML</li>
                <li>يشترط أن لا توجد مشاكل الارتباط الذاتي وعدم تجانس التباين وغيرها</li>
                <li>في حالة العينات الصغيرة، يمكن اللجوء إلى طرق تصحيح التحيز في النماذج الديناميكية مثل LSDV bias corrected</li>
            </ul>

            <h3>الصيغة الرياضية للنموذج الديناميكي البسيط</h3>
        </div>
        """, unsafe_allow_html=True)

		st.latex(r"""
        y_{it} = \alpha_i + \gamma y_{i,t-1} + \boldsymbol{x}_{it}' \boldsymbol{\beta} + \varepsilon_{it}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>نماذج البانل الديناميكية في حالة N أكبر من T</h3>
            <p>عندما يكون عدد المقاطع العرضية (N) أكبر من عدد الفترات الزمنية (T)، تظهر مشكلة التحيز في تقدير المعلمات باستخدام الطرق التقليدية. في هذه الحالة، يمكن استخدام:</p>
            <ul>
                <li>طريقة Arellano-Bond (difference GMM)</li>
                <li>طريقة Arellano-Bover/Blundell-Bond (system GMM)</li>
            </ul>

            <h3>نماذج البانل الديناميكية في حالة T أكبر من N أو كلاهما كبيرين</h3>
            <p>في هذه الحالة، يمكن استخدام:</p>
            <ul>
                <li>طريقة Mean Group (MG)</li>
                <li>طريقة Pooled Mean Group (PMG)</li>
                <li>طريقة Dynamic Fixed Effects (DFE)</li>
            </ul>

            <h3>الاختبارات المسبقة المهمة</h3>
            <ul>
                <li>اختبارات عدم تجانس الميول</li>
                <li>اختبارات cross-sectional dependence</li>
                <li>اختبارات التغير الهيكلي</li>
                <li>اختبارات الاستقرارية والتغير الهيكلي</li>
                <li>اختبارات التكامل المشترك (الجيل الأول والثاني والثالث)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني لتوضيح تحيز التقدير في النماذج الديناميكية
		fig = go.Figure()

		# إنشاء بيانات وهمية
		true_gamma = 0.7
		gamma_ols = [0.9, 0.85, 0.82, 0.79, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72]
		gamma_fe = [0.55, 0.58, 0.61, 0.63, 0.65, 0.66, 0.67, 0.68, 0.69, 0.69]
		gamma_gmm = [0.72, 0.71, 0.71, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
		t_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

		# إضافة القيمة الحقيقية
		fig.add_shape(
			type='line',
			x0=0, y0=true_gamma,
			x1=55, y1=true_gamma,
			line=dict(color='#2ecc71', width=2, dash='dash')
		)

		# إضافة تقديرات مختلفة
		fig.add_trace(go.Scatter(
			x=t_values, y=gamma_ols,
			mode='lines+markers',
			name='تقدير OLS',
			line=dict(color='#e74c3c', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t_values, y=gamma_fe,
			mode='lines+markers',
			name='تقدير Fixed Effects',
			line=dict(color='#3498db', width=2)
		))

		fig.add_trace(go.Scatter(
			x=t_values, y=gamma_gmm,
			mode='lines+markers',
			name='تقدير GMM',
			line=dict(color='#f39c12', width=2)
		))

		# إضافة تسمية للقيمة الحقيقية
		fig.add_annotation(
			x=50, y=true_gamma,
			text="القيمة الحقيقية",
			showarrow=True,
			arrowhead=1,
			ax=50, ay=-30,
			bgcolor='#2ecc71',
			bordercolor='#2ecc71',
			font=dict(color='white', size=10)
		)

		fig.update_layout(
			title="تحيز التقدير في النماذج الديناميكية حسب T",
			xaxis_title="عدد الفترات الزمنية (T)",
			yaxis_title="تقدير المعامل γ",
			height=350
		)

		st.plotly_chart(fig, use_container_width=True)

		# رسم بياني للمقارنة بين طرق التقدير المختلفة
		fig2 = go.Figure()

		methods = ['OLS', 'FE', 'Diff-GMM', 'Sys-GMM', 'LSDVC', 'MG', 'PMG', 'DFE']

		n_small_t_small = [2, 1, 4, 5, 5, 1, 3, 2]
		n_small_t_large = [2, 3, 2, 3, 4, 5, 5, 4]
		n_large_t_small = [2, 1, 5, 5, 4, 1, 3, 2]
		n_large_t_large = [3, 4, 3, 4, 4, 5, 5, 5]

		# إضافة البيانات
		fig2.add_trace(go.Bar(
			x=methods,
			y=n_small_t_small,
			name='N صغير، T صغير',
			marker_color='#3a506b'
		))

		fig2.add_trace(go.Bar(
			x=methods,
			y=n_small_t_large,
			name='N صغير، T كبير',
			marker_color='#f05454'
		))

		fig2.add_trace(go.Bar(
			x=methods,
			y=n_large_t_small,
			name='N كبير، T صغير',
			marker_color='#30475e'
		))

		fig2.add_trace(go.Bar(
			x=methods,
			y=n_large_t_large,
			name='N كبير، T كبير',
			marker_color='#7b68ee'
		))

		fig2.update_layout(
			title="مقارنة بين طرق تقدير نماذج البانل الديناميكية",
			xaxis_title="طريقة التقدير",
			yaxis_title="درجة الملاءمة (1-5)",
			height=350,
			barmode='group'
		)

		st.plotly_chart(fig2, use_container_width=True)

		# مثال على تطبيق نموذج بانل ديناميكي
		st.markdown("<h3>مثال على تطبيق نموذج بانل ديناميكي (System GMM)</h3>", unsafe_allow_html=True)
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm

        # يتطلب تثبيت حزمة linearmodels
        from linearmodels.panel import PanelOLS, FirstDifferenceOLS, RandomEffects
        from linearmodels.panel.model import PanelGMM

        # إعداد البيانات
        df = pd.read_csv('data.csv')
        df = df.set_index(['id', 'time'])

        # تحديد المتغيرات
        endog = df['y']
        exog = sm.add_constant(df[['y_lag1', 'x1', 'x2']])

        # تعريف الأدوات
        instruments = ['y_lag2', 'y_lag3', 'x1', 'x2']

        # تقدير نموذج System GMM
        model = PanelGMM.from_formula(
            formula='y ~ 1 + y_lag1 + x1 + x2',
            data=df,
            instruments=instruments
        )

        results = model.fit()
        print(results.summary)

        # اختبار صلاحية الأدوات
        sargan_test = results.sargan
        print(f"Sargan Test: {sargan_test}")
        """
		st.code(code, language='python')

# نماذج البانل الساكنة
elif choice == "نماذج البانل الساكنة":
	st.markdown("<div class='model-title'><h2>نماذج البانل الساكنة (Static Panel Models)</h2></div>",
				unsafe_allow_html=True)

	col1, col2 = st.columns([2, 1])

	with col1:
		st.markdown("""
        <div class='model-card'>
            <h3>الهدف من النموذج</h3>
            <p>دراسة التأثيرات الساكنة للمتغيرات المستقلة على المتغير التابع في إطار بيانات البانل، مع الاستفادة من البعدين المقطعي والزمني للبيانات.</p>

            <h3>الشروط الأساسية</h3>
            <ul>
                <li>نماذج البانل الساكنة التقليدية تشترط أن تكون الميول ثابتة والثوابت متغيرة</li>
                <li>يمكن استخدام نماذج البانل الساكنة في حالة N كبيرة أو T كبيرة، لكن هناك طرق حساب للتقدير تختلف حسب N وحسب T</li>
                <li>في حالة العينات الصغيرة (T و N)، يمكن استخدام bias correction للنماذج الساكنة</li>
            </ul>

            <h3>أنواع وتعديلات النموذج</h3>
            <ul>
                <li>في حالة الميول متغيرة، يمكن استخدام نماذج المعاملات المتغيرة مثل Fixed Individual Effect Variable Slopes</li>
                <li>في حالة النقاط الشاذة، يمكن استخدام One-step Robust Fixed Effect</li>
                <li>في حالة أحد المتغيرات المهمة هو Time-invariant Variables، يمكن استخدام Filtred Fixed Effect</li>
                <li>في وجود Multicolinearity، يمكن استخدام Ridge Regression للبانل</li>
                <li>في حالة مشاكل في البواقي، هناك طرق لتصحيح الانحراف المعياري مثل Driscol-Karray Methods وطرق Robust أو تغيير كامل لطرق التقدير مثل استخدام FGLS</li>
                <li>في وجود Endogeneity، يمكن استخدام Fixed or Random Instrumental Variables</li>
            </ul>

            <h3>الصيغة الرياضية للنماذج الساكنة الأساسية</h3>
        </div>
        """, unsafe_allow_html=True)

		# نموذج التأثيرات الثابتة
		st.markdown("<h4>نموذج التأثيرات الثابتة (Fixed Effects Model)</h4>", unsafe_allow_html=True)
		st.latex(r"""
        y_{it} = \alpha_i + \boldsymbol{x}_{it}' \boldsymbol{\beta} + \varepsilon_{it}
        """)

		# نموذج التأثيرات العشوائية
		st.markdown("<h4>نموذج التأثيرات العشوائية (Random Effects Model)</h4>", unsafe_allow_html=True)
		st.latex(r"""
        y_{it} = \alpha + \boldsymbol{x}_{it}' \boldsymbol{\beta} + u_i + \varepsilon_{it}
        """)

		st.markdown("""
        <div class='model-card'>
            <h3>الاختبارات المهمة في نماذج البانل الساكنة</h3>
            <ul>
                <li>اختبار Hausman: للمفاضلة بين نموذج التأثيرات الثابتة والتأثيرات العشوائية</li>
                <li>اختبار Breusch-Pagan: للمفاضلة بين نموذج التأثيرات العشوائية ونموذج الانحدار التجميعي</li>
                <li>اختبار F: للمفاضلة بين نموذج التأثيرات الثابتة ونموذج الانحدار التجميعي</li>
                <li>اختبارات Cross-sectional Dependence: مثل اختبار Pesaran CD</li>
                <li>اختبارات Heteroskedasticity: مثل اختبار Modified Wald للتأثيرات الثابتة</li>
                <li>اختبارات Serial Correlation: مثل اختبار Wooldridge للارتباط الذاتي في بيانات البانل</li>
            </ul>

            <h3>مشاكل النماذج الساكنة وطرق معالجتها</h3>
            <ul>
                <li>Heteroskedasticity: استخدام Robust Standard Errors أو FGLS</li>
                <li>Serial Correlation: استخدام Clustered Standard Errors أو نماذج FGLS</li>
                <li>Cross-sectional Dependence: استخدام Driscoll-Kraay Standard Errors أو Common Correlated Effects (CCE)</li>
                <li>Endogeneity: استخدام Instrumental Variables أو نماذج GMM</li>
                <li>Outliers: استخدام Robust Regression Methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

	with col2:
		# رسم بياني لشرح الفرق بين نماذج البانل المختلفة
		fig = go.Figure()

		# إنشاء بيانات وهمية لتوضيح الفروقات بين النماذج
		np.random.seed(42)

		# إنشاء بيانات لثلاث مجموعات
		x = np.linspace(0, 10, 20)

		# نموذج الانحدار التجميعي (نفس الميل والثابت)
		y_pooled_1 = 2 + 1.5 * x + np.random.normal(0, 1, 20)
		y_pooled_2 = 2 + 1.5 * x + np.random.normal(0, 1, 20)
		y_pooled_3 = 2 + 1.5 * x + np.random.normal(0, 1, 20)

		# نموذج التأثيرات الثابتة (نفس الميل، ثوابت مختلفة)
		y_fe_1 = 1 + 1.5 * x + np.random.normal(0, 0.7, 20)
		y_fe_2 = 3 + 1.5 * x + np.random.normal(0, 0.7, 20)
		y_fe_3 = 5 + 1.5 * x + np.random.normal(0, 0.7, 20)

		# نموذج الميول المتغيرة (ميول وثوابت مختلفة)
		y_vs_1 = 1 + 1.0 * x + np.random.normal(0, 0.5, 20)
		y_vs_2 = 3 + 1.5 * x + np.random.normal(0, 0.5, 20)
		y_vs_3 = 5 + 2.0 * x + np.random.normal(0, 0.5, 20)

		# إنشاء ثلاث رسومات بيانية منفصلة
		# 1. نموذج الانحدار التجميعي
		fig1 = go.Figure()

		fig1.add_trace(go.Scatter(
			x=x, y=y_pooled_1,
			mode='markers',
			name='المجموعة 1',
			marker=dict(color='#3a506b', size=8)
		))

		fig1.add_trace(go.Scatter(
			x=x, y=y_pooled_2,
			mode='markers',
			name='المجموعة 2',
			marker=dict(color='#f05454', size=8)
		))

		fig1.add_trace(go.Scatter(
			x=x, y=y_pooled_3,
			mode='markers',
			name='المجموعة 3',
			marker=dict(color='#30475e', size=8)
		))

		# إضافة خط الانحدار
		all_x = np.concatenate([x, x, x])
		all_y = np.concatenate([y_pooled_1, y_pooled_2, y_pooled_3])
		coef = np.polyfit(all_x, all_y, 1)
		line = coef[0] * np.linspace(0, 10, 100) + coef[1]

		fig1.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=line,
			mode='lines',
			name='خط الانحدار المجمع',
			line=dict(color='#7b68ee', width=3)
		))

		fig1.update_layout(
			title="نموذج الانحدار التجميعي",
			xaxis_title="X",
			yaxis_title="Y",
			height=200
		)

		# 2. نموذج التأثيرات الثابتة
		fig2 = go.Figure()

		fig2.add_trace(go.Scatter(
			x=x, y=y_fe_1,
			mode='markers',
			name='المجموعة 1',
			marker=dict(color='#3a506b', size=8)
		))

		fig2.add_trace(go.Scatter(
			x=x, y=y_fe_2,
			mode='markers',
			name='المجموعة 2',
			marker=dict(color='#f05454', size=8)
		))

		fig2.add_trace(go.Scatter(
			x=x, y=y_fe_3,
			mode='markers',
			name='المجموعة 3',
			marker=dict(color='#30475e', size=8)
		))

		# إضافة خطوط انحدار منفصلة بنفس الميل
		slope = 1.5

		fig2.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=slope * np.linspace(0, 10, 100) + 1,
			mode='lines',
			name='خط المجموعة 1',
			line=dict(color='#3a506b', width=3)
		))

		fig2.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=slope * np.linspace(0, 10, 100) + 3,
			mode='lines',
			name='خط المجموعة 2',
			line=dict(color='#f05454', width=3)
		))

		fig2.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=slope * np.linspace(0, 10, 100) + 5,
			mode='lines',
			name='خط المجموعة 3',
			line=dict(color='#30475e', width=3)
		))

		fig2.update_layout(
			title="نموذج التأثيرات الثابتة",
			xaxis_title="X",
			yaxis_title="Y",
			height=200,
			showlegend=False
		)

		# 3. نموذج الميول المتغيرة
		fig3 = go.Figure()

		fig3.add_trace(go.Scatter(
			x=x, y=y_vs_1,
			mode='markers',
			name='المجموعة 1',
			marker=dict(color='#3a506b', size=8)
		))

		fig3.add_trace(go.Scatter(
			x=x, y=y_vs_2,
			mode='markers',
			name='المجموعة 2',
			marker=dict(color='#f05454', size=8)
		))

		fig3.add_trace(go.Scatter(
			x=x, y=y_vs_3,
			mode='markers',
			name='المجموعة 3',
			marker=dict(color='#30475e', size=8)
		))

		# إضافة خطوط انحدار منفصلة بميول مختلفة
		fig3.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=1.0 * np.linspace(0, 10, 100) + 1,
			mode='lines',
			name='خط المجموعة 1',
			line=dict(color='#3a506b', width=3)
		))

		fig3.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=1.5 * np.linspace(0, 10, 100) + 3,
			mode='lines',
			name='خط المجموعة 2',
			line=dict(color='#f05454', width=3)
		))

		fig3.add_trace(go.Scatter(
			x=np.linspace(0, 10, 100), y=2.0 * np.linspace(0, 10, 100) + 5,
			mode='lines',
			name='خط المجموعة 3',
			line=dict(color='#30475e', width=3)
		))

		fig3.update_layout(
			title="نموذج الميول المتغيرة",
			xaxis_title="X",
			yaxis_title="Y",
			height=200,
			showlegend=False
		)

		# عرض الرسومات البيانية
		st.plotly_chart(fig1, use_container_width=True)
		st.plotly_chart(fig2, use_container_width=True)
		st.plotly_chart(fig3, use_container_width=True)

		# مثال على تطبيق نماذج البانل الساكنة
		st.markdown("<h3>مثال على تطبيق نماذج البانل الساكنة</h3>", unsafe_allow_html=True)
		code = """
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

        # إعداد البيانات
        df = pd.read_csv('data.csv')
        df = df.set_index(['id', 'time'])

        # 1. نموذج الانحدار التجميعي
        pooled_model = PooledOLS.from_formula('y ~ 1 + x1 + x2', data=df)
        pooled_results = pooled_model.fit()

        # 2. نموذج التأثيرات الثابتة
        fe_model = PanelOLS.from_formula('y ~ 1 + x1 + x2 + EntityEffects', data=df)
        fe_results = fe_model.fit()

        # 3. نموذج التأثيرات العشوائية
        re_model = RandomEffects.from_formula('y ~ 1 + x1 + x2', data=df)
        re_results = re_model.fit()

        # اختبار Hausman
        hausman_statistic = fe_results.test_against(re_results)

        # عرض النتائج
        print(fe_results.summary)
        print(f"Hausman Test: {hausman_statistic}")
        """
		st.code(code, language='python')

# المتناقضات في الدراسات العربية
elif choice == "المتناقضات في الدراسات العربية":
	st.markdown("<div class='model-title'><h2>المتناقضات في الدراسات العربية</h2></div>", unsafe_allow_html=True)

	st.markdown("""
    <div class='model-card' style='background-color: #ffebee;'>
        <h3>أهم المتناقضات في الدراسات العربية</h3>
        <ul>
            <li>لا يجوز استخدام اختبار جوهانسون في وجود تغيرات هيكلية</li>
            <li>لا يمكن أن نجمع بين ARDL و VAR في دراسة واحدة، لأن الأول يعتمد على معادلة واحدة والآخر يعتمد على نظام من المعادلات</li>
            <li>لا يمكن الجمع بين اختبار جوهانسون و Bounds Test، لأن جوهانسون يختبر العلاقة التبادلية أما Bounds Test فيختبر العلاقة في اتجاه واحد</li>
            <li>لا يمكن الجمع بين اختبارات الجيل الأول والثاني في البانل للتكامل المشترك أو جذر الوحدة، لأن في وجود أو غياب Cross-sectional Dependence سيبقى لنا اختبار واحد فقط إما من الجيل الأول أو الثاني</li>
            <li>لا يمكن الجمع بين ARDL-PMG و ARDL-CS، لأن وجود أو غياب Cross-sectional Dependence سيبقي لنا نموذج واحد فقط</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	# إضافة رسم بياني توضيحي للتناقضات
	fig = go.Figure()

	contradictions = [
		"استخدام اختبار جوهانسون مع تغيرات هيكلية",
		"الجمع بين ARDL و VAR في نفس الدراسة",
		"الجمع بين اختبار جوهانسون و Bounds Test",
		"الجمع بين اختبارات الجيل الأول والثاني للتكامل المشترك",
		"الجمع بين ARDL-PMG و ARDL-CS"
	]

	frequency = [68, 45, 72, 53, 40]
	severity = [4, 3, 5, 4, 3]

	# تحويل الحجم إلى نطاق مناسب
	bubble_size = [s * 15 for s in severity]

	fig.add_trace(go.Scatter(
		x=frequency,
		y=[1, 2, 3, 4, 5],
		mode='markers',
		marker=dict(
			size=bubble_size,
			color=['#e74c3c', '#e67e22', '#c0392b', '#d35400', '#e74c3c'],
			opacity=0.8,
			line=dict(color='white', width=1)
		),
		text=contradictions,
		hoverinfo='text'
	))

	for i, txt in enumerate(contradictions):
		fig.add_annotation(
			x=frequency[i],
			y=i + 1,
			text=txt,
			showarrow=False,
			font=dict(size=10),
			xshift=15,
			align='left'
		)

	fig.update_layout(
		title="تكرار المتناقضات في الدراسات العربية",
		xaxis_title="تكرار الظهور في الدراسات",
		yaxis=dict(
			showticklabels=False,
			showgrid=False
		),
		height=400,
		showlegend=False
	)

	st.plotly_chart(fig, use_container_width=True)

	# نصائح لتجنب المتناقضات
	st.markdown("""
    <div class='model-card' style='background-color: #e8f5e9;'>
        <h3>نصائح لتجنب المتناقضات في الدراسات الاقتصادية القياسية</h3>
        <ol>
            <li><strong>فهم أساسيات النموذج:</strong> فهم الافتراضات الأساسية والشروط اللازمة لكل نموذج قبل تطبيقه.</li>
            <li><strong>اختيار النموذج المناسب:</strong> اختيار النموذج الذي يتناسب مع طبيعة البيانات وأهداف الدراسة.</li>
            <li><strong>إجراء الاختبارات التشخيصية:</strong> التحقق من صلاحية النموذج من خلال الاختبارات التشخيصية المناسبة.</li>
            <li><strong>مراعاة خصائص البيانات:</strong> الانتباه إلى خصائص البيانات مثل الاستقرارية والتغيرات الهيكلية.</li>
            <li><strong>تجنب الجمع بين النماذج المتعارضة:</strong> تجنب استخدام نماذج ذات افتراضات متعارضة في نفس الدراسة.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ملاحظات عامة
elif choice == "ملاحظات عامة":
	st.markdown("<div class='model-title'><h2>ملاحظات عامة</h2></div>", unsafe_allow_html=True)

	st.markdown("""
    <div class='model-card'>
        <ul>
            <li>هذا المخطط يتكلم عن أهم النماذج التفسيرية في الدراسات العربية وليست التنبؤية.</li>
            <li>هذه الشروط بصفة عامة وليست مفصلة، لأن الشروط المفصلة تحتاج مخطط لكل نموذج على حدى.</li>
            <li>دائماً عندنا شروط متعلقة بالبواقي وهي أن تكون خالية من المشاكل، وهذه خاصية مشتركة بين كل النماذج في المخطط.</li>
            <li>المعيار الأهم لاختيار نموذج معين هو هل أهدافه تتوافق مع أهداف الدراسة وهل يستطيع أن يجيب عن إشكالية البحث، وبعدها نتكلم عن الجزئيات.</li>
            <li>من شروط تعلم أي نموذج هو التركيز على الأهداف والشروط والتمرن على التطبيق ومعرفة الانتقادات والعمل على البدائل.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	# إضافة رسم بياني للعلاقة بين معايير اختيار النموذج
	fig = go.Figure()

	criteria = [
		"توافق النموذج مع أهداف الدراسة",
		"قدرة النموذج على الإجابة عن إشكالية البحث",
		"تحقق شروط تطبيق النموذج",
		"توافر البيانات اللازمة",
		"سهولة التفسير والتحليل"
	]

	importance = [5, 4.8, 4.2, 3.5, 3.2]

	fig.add_trace(go.Bar(
		x=importance,
		y=criteria,
		orientation='h',
		marker=dict(
			color=['#3a506b', '#3a506b', '#3a506b', '#3a506b', '#3a506b'],
			colorscale=[[0, '#f05454'], [1, '#3a506b']],
			line=dict(color='white', width=1)
		)
	))

	fig.update_layout(
		title="معايير اختيار النموذج القياسي المناسب (حسب الأهمية)",
		xaxis_title="درجة الأهمية",
		yaxis=dict(
			title="",
			autorange="reversed"
		),
		height=350
	)

	st.plotly_chart(fig, use_container_width=True)

	# الخاتمة والتوصيات
	st.markdown("""
    <div class='model-card' style='background-color: #e3f2fd;'>
        <h3>توصيات لاستخدام النماذج القياسية</h3>
        <ol>
            <li>ضرورة فهم الأسس النظرية والافتراضات الأساسية للنماذج القياسية قبل تطبيقها.</li>
            <li>أهمية اختيار النموذج المناسب وفقاً لطبيعة البيانات وأهداف الدراسة.</li>
            <li>ضرورة إجراء الاختبارات التشخيصية للتحقق من صلاحية النموذج.</li>
            <li>تجنب استخدام النماذج المتناقضة في نفس الدراسة.</li>
            <li>الاطلاع المستمر على التطورات الحديثة في مجال النمذجة القياسية.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class='note'>
        <p style='text-align: center;'>إعداد: Merwan Roudane</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 - النماذج القياسية الأكثر شهرة عربياً</p>", unsafe_allow_html=True)
