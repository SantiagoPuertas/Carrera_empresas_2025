import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np
import plotly.graph_objects as go
import unicodedata
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.io as pio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt





# ------------------------
# FUNCIONES
# ------------------------
def segundos_a_hms_str(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"



def normalizar(texto):
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = " ".join(texto.split())
    return texto

def draw_text_shadow(draw, xy, text, font, fill, shadow=(0, 0, 0), offset=3):
    x, y = xy
    draw.text((x + offset, y + offset), text, font=font, fill=shadow, anchor="mm")
    draw.text((x, y), text, font=font, fill=fill, anchor="mm")



def plotly_to_pil(fig, width=900, height=450):
    img_bytes = pio.to_image(
        fig,
        format="png",
        width=width,
        height=height,
        scale=2
    )
    return Image.open(io.BytesIO(img_bytes))

def generar_tarjeta_runner(
    nombre,
    distancia,
    sexo,
    tiempo,
    percentil,
    puesto_empresa,
    empresa,
    img_cdf=None
):
    # Tamaño Instagram-friendly (4:5)
    W, H = 1080, 1350

    fondo = Image.open("fondo_tarjeta.png").resize((W, H))
    img = fondo.copy()
    draw = ImageDraw.Draw(img)

    # Fuentes (fallback seguro)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
        font_name_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 86)
        font_name_small = ImageFont.truetype("DejaVuSans-Bold.ttf", 70)
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 88)
        font_text = ImageFont.truetype("DejaVuSans.ttf", 46)
    except:
        font_title = font_name_big = font_name_small = font_big = font_text = ImageFont.load_default()

    # Colores
    gold = "#ffd700"
    azul = "#205AEB"
    gris = "#A0A0A0"
    negro = "#000000"

    # ---------- LAYOUT ----------
    y = 160  # margen superior seguro

    # Título
    draw.text(
        (W // 2, y),
        "Carrera de las Empresas 2025",
        font=font_title,
        fill=gris,
        anchor="mm"
    )
    y += 130

    # Nombre (ajuste dinámico)
    font_name = font_name_big if len(nombre) <= 22 else font_name_small

    draw.text(
        (W // 2, y),
        nombre,
        font=font_name,
        fill=azul,
        anchor="mm"
    )
    y += 110

    # Subtítulo
    draw.text(
        (W // 2, y),
        f"{sexo} · {distancia}",
        font=font_text,
        fill=gris,
        anchor="mm"
    )
    y += 130

    # Tiempo
    draw.text(
        (W // 2, y),
        f"{tiempo}",
        font=font_big,
        fill=negro,
        anchor="mm"
    )
    y += 140

    # -------- CDF --------
    if img_cdf is not None:
        cdf_width = int(W * 0.9)
        cdf_height = int(cdf_width * img_cdf.height / img_cdf.width)

        img_cdf_resized = img_cdf.resize((cdf_width, cdf_height))

        img.paste(
            img_cdf_resized,
            (W // 2 - cdf_width // 2, y)
        )

        y += cdf_height + 40


    # Percentil
    draw.text(
        (W // 2, y),
        f"Percentil: {percentil:.1f} %",
        font=font_text,
        fill=negro,
        anchor="mm"
    )
    y += 80

    # Puesto empresa
    draw.text(
        (W // 2, y),
        f"Puesto en {empresa}: {puesto_empresa}",
        font=font_text,
        fill=gold,
        anchor="mm"
    )

    # Footer
    draw.text(
        (W // 2, H - 50),
        "https://dashcarreraempresas2025.streamlit.app/",
        font=font_text,
        fill=negro,
        anchor="mm"
    )

    # Exportar a bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def generar_cdf_matplotlib(subset, runner_time):
    x = np.sort(subset["tiempo_segundos"].values)
    y = np.arange(1, len(x) + 1) / len(x) * 100

    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)

    ax.plot(x, y, color="#205AEB", linewidth=2)
    ax.axvline(runner_time, color="red", linestyle="--", linewidth=2)

    percentil = (x < runner_time).mean() * 100
    ax.axhline(percentil, color="red", linestyle=":", linewidth=1.5)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 100)

    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Percentil", fontsize=9)

    ticks = np.linspace(x.min(), x.max(), 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([segundos_a_hms_str(v) for v in ticks], fontsize=8)

    ax.grid(alpha=0.3)
    ax.set_title("CDF", fontsize=10)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


# ------------------------
# CARGA DE DATOS
# ------------------------

#df = pd.read_excel(
#    "clasificacion_carrera_empresas_2025.xlsx",
#    sheet_name="total"
#)



# -------------------------
# CARGA DE DATOS DESDE GOOGLE SHEETS
# -------------------------

@st.cache_data(ttl=3600)  # 1 hora
def cargar_datos():
    url = st.secrets["DATA_URL"]
    return pd.read_excel(url, sheet_name="total")

# ------------------------cargamos la data
df = cargar_datos()
# ------------------------

# Tiempo en segundos (si no existe ya)
if "tiempo_segundos" not in df.columns:
    df["tiempo_segundos"] = df["tiempo"].apply(
        lambda t: t.hour*3600 + t.minute*60 + t.second
    )


# ------------------------
# Selectores
# ------------------------

st.sidebar.header("Filtros")

sexo_sel = st.sidebar.selectbox(
    "Sexo",
    ["Masculino", "Femenino", "Ambos"],
    key="sexo_global"
)

dist_sel = st.sidebar.selectbox(
    "Distancia",
    ["5km", "10km"],
    key="distancia_global"
)


# -------------------------
# IDENTIFICACIÓN DEL CORREDOR
# -------------------------
st.sidebar.header("Identifícate")

nombre_input = st.sidebar.text_input(
    "Introduce tu nombre completo",
    value=""
).strip()

if nombre_input == "":
    st.info("Introduce tu nombre para ver tu análisis individual.")
    st.stop()

# Normalizar nombres
if "nombre_norm" not in df.columns:
    df["nombre_norm"] = df["nombre"].apply(normalizar)

nombre_norm = normalizar(nombre_input)
tokens = nombre_norm.split()

matches = df.copy()
for token in tokens:
    matches = matches[matches["nombre_norm"].str.contains(token, na=False)]

if matches.empty:
    st.error("No se ha encontrado ningún corredor con ese nombre.")
    st.stop()

if len(matches) > 1:
    st.warning(
        "Se han encontrado varios corredores con un nombre similar. "
        "Por favor, escribe el nombre completo."
    )
    st.dataframe(
        matches[["nombre", "empresa", "Distancia", "Categoria"]],
        use_container_width=True
    )
    st.stop()

runner = matches.iloc[0]




# ------------------------
# UI
# ------------------------
st.title("Análisis individual de rendimiento")




modo = st.radio(
    "Compararte contra:",
    [
        "Mi categoría (sexo + distancia)",
        "General (misma distancia)",
        "Mi empresa"
    ]
)





# ------------------------
# DATOS DEL CORREDOR
# ------------------------




st.subheader("Datos del corredor")
st.write({
    "Nombre": runner["nombre"],
    "Distancia": runner["Distancia"],
    "Sexo": runner["Categoria"],
    "Empresa": runner["empresa"],
    "Tiempo": runner["tiempo"]
})

# ------------------------
# CONSTRUCCIÓN DEL SUBSET
# ------------------------
if modo == "Mi categoría (sexo + distancia)":
    subset = df[
        (df["Distancia"] == runner["Distancia"]) &
        (df["Categoria"] == runner["Categoria"])
    ]
    titulo = f"{runner['Categoria']} – {runner['Distancia']}"

elif modo == "General (misma distancia)":
    subset = df[df["Distancia"] == runner["Distancia"]]
    titulo = f"General – {runner['Distancia']}"

else:  # empresa
    subset = df[
        (df["empresa"] == runner["empresa"]) &
        (df["Distancia"] == runner["Distancia"])
    ]
    titulo = f"Empresa – {runner['empresa']}"


    

# ------------------------
# PERCENTIL
# ------------------------
percentil = (
    (subset["tiempo_segundos"] < runner["tiempo_segundos"]).mean()
) * 100

# =========================
# GRAFICO SUAVIZADO (CONTEO REAL)
# =========================

x = subset["tiempo_segundos"].values
n = len(x)

# Configuración bins
nbins = 40
bin_width = (x.max() - x.min()) / nbins

# KDE
kde = gaussian_kde(x)
x_kde = np.linspace(x.min(), x.max(), 500)
y_kde = kde(x_kde) * n * bin_width  # ESCALADO A CONTEO

fig = go.Figure()

#histrograma
fig.add_trace(
    go.Histogram(
        x=x,
        nbinsx=nbins,
        marker=dict(
            color="rgba(100, 180, 255, 0.4)",
            line=dict(width=0)
        ),
        name="Distribución",
        customdata=[segundos_a_hms_str(v) for v in x],
        hovertemplate=(
            "Tiempo: %{customdata}<br>"
            "Corredores: %{y}<extra></extra>"
        )
    )
)


# Curva KDE escalada
fig.add_trace(
    go.Scatter(
        x=x_kde,
        y=y_kde,
        mode="lines",
        line=dict(color="white", width=3),
        name="Densidad suavizada",
        customdata=[segundos_a_hms_str(v) for v in x_kde],
        hovertemplate=(
            "Tiempo: %{customdata}<br>"
            "Corredores (estimado): %{y:.0f}<extra></extra>"
        )
    )
)


# Resaltar TU posición
mi_tiempo = runner["tiempo_segundos"]

fig.add_vrect(
    x0=mi_tiempo - 30,
    x1=mi_tiempo + 30,
    fillcolor="rgba(255, 80, 80, 0.35)",
    layer="below",
    line_width=0,
)

fig.add_vline(
    x=mi_tiempo,
    line_width=3,
    line_dash="dash",
    line_color="red",
    annotation_text="Tú",
    annotation_position="top"
)

# Eje X HH:MM:SS
tickvals = np.linspace(x.min(), x.max(), 7)
ticktext = [segundos_a_hms_str(v) for v in tickvals]

fig.update_xaxes(
    title="Tiempo (HH:MM:SS)",
    tickvals=tickvals,
    ticktext=ticktext
)

fig.update_yaxes(
    title="Número de corredores"
)

fig.update_layout(
    title=f"Distribución de tiempos · {titulo}",
    bargap=0.02,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)



# ------------------------
# MÉTRICAS
# ------------------------
st.metric(
    label="Percentil",
    value=f"{percentil:.1f} %"
)

st.metric(
    label="Posición aproximada",
    value=f"{int(percentil/100 * len(subset))} / {len(subset)}"
)


# =========================
# PERCENTILES
# =========================

percentiles = {
    "P25": subset["tiempo_segundos"].quantile(0.25),
    "P50 (Mediana)": subset["tiempo_segundos"].quantile(0.50),
    "P75": subset["tiempo_segundos"].quantile(0.75),
}

st.subheader("Percentiles del grupo de comparación")

cols = st.columns(len(percentiles))
for col, (label, value) in zip(cols, percentiles.items()):
    col.metric(
        label,
        segundos_a_hms_str(value)
    )

# =========================
# EMPRESA vs GLOBAL (% DE CORREDORES POR FRANJA)
# =========================

st.subheader("Empresa vs Global · % de corredores por franja")

# Dataset base (misma distancia y sexo)
df_global = df[
    (df["Distancia"] == runner["Distancia"]) &
    (df["Categoria"] == runner["Categoria"])
]

df_empresa = df_global[df_global["empresa"] == runner["empresa"]]

fig_emp = go.Figure()

# -------------------------
# BINS COMUNES
# -------------------------
x_all = df_global["tiempo_segundos"].values
bin_size = 60  # segundos (1 minuto)
bins = np.arange(x_all.min(), x_all.max() + bin_size, bin_size)

edges = bins
centros = (edges[:-1] + edges[1:]) / 2

# -------------------------
# GLOBAL (% por franja)
# -------------------------
counts_g, _ = np.histogram(df_global["tiempo_segundos"].values, bins=bins)
pct_g = counts_g / counts_g.sum() * 100

hover_global = [
    f"Intervalo: {segundos_a_hms_str(edges[i])} – {segundos_a_hms_str(edges[i+1])}<br>"
    f"% corredores: {pct_g[i]:.2f} %"
    for i in range(len(pct_g))
]

fig_emp.add_trace(
    go.Bar(
        x=centros,
        y=pct_g,
        hovertext=hover_global,
        hoverinfo="text",
        name="Global",
        marker=dict(color="rgba(120,120,120,0.45)")
    )
)

# -------------------------
# EMPRESA (% por franja)
# -------------------------
if not df_empresa.empty:
    counts_e, _ = np.histogram(df_empresa["tiempo_segundos"].values, bins=bins)
    pct_e = counts_e / counts_e.sum() * 100

    hover_empresa = [
        f"Intervalo: {segundos_a_hms_str(edges[i])} – {segundos_a_hms_str(edges[i+1])}<br>"
        f"% corredores: {pct_e[i]:.2f} %"
        for i in range(len(pct_e))
    ]

    fig_emp.add_trace(
        go.Bar(
            x=centros,
            y=pct_e,
            hovertext=hover_empresa,
            hoverinfo="text",
            name=runner["empresa"],
            marker=dict(color="rgba(0,180,255,0.65)")
        )
    )

# -------------------------
# EJES Y LAYOUT
# -------------------------
tickvals = np.linspace(edges.min(), edges.max(), 7)
ticktext = [segundos_a_hms_str(v) for v in tickvals]

fig_emp.update_xaxes(
    title="Tiempo (HH:MM:SS)",
    tickvals=tickvals,
    ticktext=ticktext
)

fig_emp.update_yaxes(
    title="% de corredores",
    ticksuffix=" %",
    range=[0, max(pct_g.max(), pct_e.max() if not df_empresa.empty else 0) * 1.1]
)

fig_emp.update_layout(
    title=f"Empresa vs Global · % de corredores por franja · {runner['Categoria']} {runner['Distancia']}",
    barmode="overlay",
    bargap=0.05,
    template="plotly_dark"
)

st.plotly_chart(fig_emp, use_container_width=True)





# =========================
# RANKING DENTRO DE EMPRESA
# =========================

st.subheader("Ranking dentro de la empresa")

df_empresa_rank = df[
    (df["empresa"] == runner["empresa"]) &
    (df["Distancia"] == runner["Distancia"]) &
    (df["Categoria"] == runner["Categoria"])
].sort_values("tiempo_segundos").reset_index(drop=True)

df_empresa_rank["puesto_empresa"] = df_empresa_rank.index + 1

mi_fila = df_empresa_rank[df_empresa_rank["nombre"] == runner["nombre"]].iloc[0]

st.markdown(
    f"""
    **Empresa:** {runner['empresa']}  
    **Distancia:** {runner['Distancia']}  
    **Sexo:** {runner['Categoria']}  

    🏅 **Tu puesto:** {int(mi_fila['puesto_empresa'])} de {len(df_empresa_rank)}
    """
)

# -------------------------
# TABLA: TOP EMPRESA + TÚ
# -------------------------
# =========================
# RANKING DENTRO DE LA EMPRESA
# =========================

df_empresa_rank = (
    df[
        (df["empresa"] == runner["empresa"]) &
        (df["Distancia"] == runner["Distancia"]) &
        (df["Categoria"] == runner["Categoria"])
    ]
    .sort_values("tiempo_segundos")
    .reset_index(drop=True)
)

# Puesto dentro de la empresa
df_empresa_rank["puesto_empresa"] = df_empresa_rank.index + 1

# Puesto general (misma distancia, ambos sexos)
df_empresa_rank["puesto_absoluto_distancia"] = (
    df_empresa_rank["tiempo_segundos"]
    .apply(
        lambda t: (
            df[
                (df["Distancia"] == runner["Distancia"]) &
                (df["tiempo_segundos"] <= t)
            ]
            .shape[0]
        )
    )
)

# Puesto en categoría (sexo + distancia)
df_empresa_rank["puesto_categoria"] = (
    df_empresa_rank["tiempo_segundos"]
    .apply(
        lambda t: (
            df[
                (df["Distancia"] == runner["Distancia"]) &
                (df["Categoria"] == runner["Categoria"]) &
                (df["tiempo_segundos"] <= t)
            ]
            .shape[0]
        )
    )
)

top_n = 10

df_mostrar = pd.concat([
    df_empresa_rank.head(top_n),
    df_empresa_rank[df_empresa_rank["nombre"] == runner["nombre"]]
]).drop_duplicates()

df_mostrar["Tiempo"] = df_mostrar["tiempo_segundos"].apply(segundos_a_hms_str)

df_mostrar = df_mostrar[[
    "puesto_empresa",
    "puesto_absoluto_distancia",
    "puesto_categoria",
    "nombre",
    "Tiempo"
]]

df_mostrar = df_mostrar.rename(columns={
    "puesto_empresa": "Puesto empresa",
    "puesto_absoluto_distancia": "Puesto general",
    "puesto_categoria": "Puesto categoría"
})

st.dataframe(
    df_mostrar.sort_values("Puesto empresa"),
    use_container_width=True
)



# =========================
# CDF INTERACTIVA
# =========================

st.subheader("Curva acumulada (CDF)")

# Dataset de comparación (el mismo subset que uses arriba)
cdf_df = subset.sort_values("tiempo_segundos").reset_index(drop=True)

cdf_df["percentil"] = (cdf_df.index + 1) / len(cdf_df) * 100

fig_cdf = go.Figure()

# Curva CDF
fig_cdf.add_trace(
    go.Scatter(
        x=cdf_df["tiempo_segundos"],
        y=cdf_df["percentil"],
        mode="lines",
        line=dict(width=3),
        name="CDF",
        customdata=[
            segundos_a_hms_str(v) for v in cdf_df["tiempo_segundos"]
        ],
        hovertemplate=(
            "Tiempo: %{customdata}<br>"
            "Percentil: %{y:.1f} %"
            "<extra></extra>"
        )
    )
)


# Línea vertical (tu tiempo)
fig_cdf.add_vline(
    x=runner["tiempo_segundos"],
    line_width=3,
    line_dash="dash",
    line_color="red"
)

# Línea horizontal (tu percentil)
mi_percentil = (
    (cdf_df["tiempo_segundos"] < runner["tiempo_segundos"]).mean()
) * 100

fig_cdf.add_hline(
    y=mi_percentil,
    line_width=2,
    line_dash="dot",
    line_color="red"
)

# Eje X en HH:MM:SS
tickvals = np.linspace(
    cdf_df["tiempo_segundos"].min(),
    cdf_df["tiempo_segundos"].max(),
    7
)
ticktext = [segundos_a_hms_str(v) for v in tickvals]

fig_cdf.update_xaxes(
    title="Tiempo (HH:MM:SS)",
    tickvals=tickvals,
    ticktext=ticktext
)

fig_cdf.update_yaxes(
    title="Percentil",
    range=[0, 100]
)

fig_cdf.update_layout(
    title=f"CDF · {titulo}",
    template="plotly_dark"
)

st.plotly_chart(fig_cdf, use_container_width=True)

st.metric(
    "Tu percentil",
    f"{mi_percentil:.1f} %"
)





# =========================
# COMPARACIÓN ENTRE EMPRESAS
# =========================

base_df = df[df["Distancia"] == dist_sel]

if sexo_sel != "Ambos":
    base_df = base_df[base_df["Categoria"] == sexo_sel]

# ---------
# FILTROS
# ---------
empresas = sorted(df["empresa"].dropna().unique())

empresas_sel = st.multiselect(
    "Selecciona empresas",
    empresas,
    default=[runner["empresa"]] if "runner" in locals() else []
)


if empresas_sel:

    # -------------------------
    # DATASET BASE
    # -------------------------
    base_df = df[df["Distancia"] == dist_sel]

    if sexo_sel != "Ambos":
        base_df = base_df[base_df["Categoria"] == sexo_sel]

    # -------------------------
    # BINS COMUNES
    # -------------------------
    x_all = base_df["tiempo_segundos"].values
    bin_size = 60  # segundos
    bins = np.arange(x_all.min(), x_all.max() + bin_size, bin_size)
    centros = (bins[:-1] + bins[1:]) / 2

    fig_comp = go.Figure()

    # -------------------------
    # CADA EMPRESA
    # -------------------------
    for emp in empresas_sel:
        subset_emp = base_df[base_df["empresa"] == emp]

        if subset_emp.empty:
            continue

        counts, _ = np.histogram(subset_emp["tiempo_segundos"].values, bins=bins)

        hover_text = [
            f"Empresa: {emp}<br>"
            f"Intervalo: {segundos_a_hms_str(bins[i])} – {segundos_a_hms_str(bins[i+1])}<br>"
            f"Corredores: {counts[i]}"
            for i in range(len(counts))
        ]

        fig_comp.add_trace(
            go.Bar(
                x=centros,
                y=counts,
                hovertext=hover_text,
                hoverinfo="text",
                name=emp,
                opacity=0.65
            )
        )

    # -------------------------
    # EJES Y LAYOUT
    # -------------------------
    tickvals = np.linspace(bins.min(), bins.max(), 7)
    ticktext = [segundos_a_hms_str(v) for v in tickvals]

    fig_comp.update_xaxes(
        title="Tiempo (HH:MM:SS)",
        tickvals=tickvals,
        ticktext=ticktext
    )

    fig_comp.update_yaxes(
        title="Número de corredores"
    )

    titulo_sexo = sexo_sel if sexo_sel != "Ambos" else "Masculino + Femenino"

    fig_comp.update_layout(
        title=f"Comparación de empresas · {titulo_sexo} · {dist_sel}",
        barmode="overlay",
        bargap=0.05,
        template="plotly_dark"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("Selecciona al menos una empresa para ver la comparación.")



# =========================
# DISTRIBUCIONES GENERALES
# =========================


st.header("Distribuciones generales")

if sexo_sel == "Ambos":
    subset_gen = df[df["Distancia"] == dist_sel]
else:
    subset_gen = df[
        (df["Categoria"] == sexo_sel) &
        (df["Distancia"] == dist_sel)
    ]

subset_gen = df[
    (df["Categoria"] == sexo_sel) &
    (df["Distancia"] == dist_sel)
]

# =========================
# HISTOGRAMA MANUAL
# =========================

x = subset_gen["tiempo_segundos"].values

bin_size = 60  # segundos (1 minuto)
bins = np.arange(x.min(), x.max() + bin_size, bin_size)

counts, edges = np.histogram(x, bins=bins)

# Centro de cada bin
centros = (edges[:-1] + edges[1:]) / 2

# Texto hover en HH:MM:SS (intervalo real)
hover_text = [
    f"Intervalo: {segundos_a_hms_str(edges[i])} – {segundos_a_hms_str(edges[i+1])}<br>"
    f"Corredores: {counts[i]}"
    for i in range(len(counts))
]

fig_gen = go.Figure()

fig_gen.add_trace(
    go.Bar(
        x=centros,
        y=counts,
        marker=dict(color="rgba(150,200,255,0.6)"),
        hovertext=hover_text,
        hoverinfo="text",
        name="Corredores"
    )
)

# ---- EJE X HH:MM:SS ----
tickvals = np.linspace(centros.min(), centros.max(), 7)
ticktext = [segundos_a_hms_str(v) for v in tickvals]

fig_gen.update_xaxes(
    title="Tiempo (HH:MM:SS)",
    tickvals=tickvals,
    ticktext=ticktext
)

fig_gen.update_yaxes(
    title="Número de corredores"
)

fig_gen.update_layout(
    title=f"Distribución general · {sexo_sel} {dist_sel}",
    bargap=0.05,
    template="plotly_dark"
)

st.plotly_chart(fig_gen, use_container_width=True)




# =========================
# compartir resultado
# =========================
img_cdf = generar_cdf_matplotlib(subset, runner["tiempo_segundos"])

st.subheader("📸 Comparte tu resultado")

img_buffer = generar_tarjeta_runner(
    nombre=runner["nombre"],
    distancia=runner["Distancia"],
    sexo=runner["Categoria"],
    tiempo=segundos_a_hms_str(runner["tiempo_segundos"]),
    percentil=mi_percentil,
    puesto_empresa=int(
        df_empresa_rank.loc[
            df_empresa_rank["nombre"] == runner["nombre"],
            "puesto_empresa"
        ].values[0]
    ),
    empresa=runner["empresa"],
    img_cdf=img_cdf
)

st.download_button(
    "📸 Descargar tarjeta para redes",
    data=img_buffer,
    file_name=f"{runner['nombre'].replace(' ', '_')}_carrera_empresas_2025.png",
    mime="image/png"
)

st.caption(
    "💡 Descárgala y compártela en Instagram, Twitter o LinkedIn"
)



st.caption(
    "Los datos utilizados no se publican ni se distribuyen. "
    "Esta aplicación es solo una herramienta de análisis individual."
)
