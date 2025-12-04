import gradio as gr
import joblib
import pandas as pd

# Model yükleme
model = joblib.load("lgb_optimized_pubg.pkl")

def predict_winplace(kills, damageDealt, walkDistance, rideDistance, 
                     weaponsAcquired, playersInGroup, playersInMatch):

    # --- Giriş sınırlarını kontrol et ---
    if kills > 99:
        return 0, "<div style='color:red; font-size:22px;'>HATA: Kill sayısı 99'dan büyük olamaz.</div>"

    if playersInMatch > 100:
        return 0, "<div style='color:red; font-size:22px;'>HATA: Maçtaki oyuncu sayısı 100'den büyük olamaz.</div>"

    # --- playersInGroup -> matchType otomatik belirleme ---
    if playersInGroup == 1:
        matchType = 0  # Solo
        matchTypeName = "Solo"
    elif playersInGroup == 2:
        matchType = 2  # Duo
        matchTypeName = "Duo"
    else:  # 3 veya 4
        matchType = 3  # Squad
        matchTypeName = "Squad"

    # --- otomatik hesaplanan özellikler ---
    totalDistance = walkDistance + rideDistance
    walkRideRatio = walkDistance / (rideDistance + 1)
    damagePerKill = damageDealt / (kills + 1)
    combatScore = kills * 0.7 + damageDealt * 0.3
    playerAggression = walkDistance * 0.5 + kills * 1.5 + damageDealt * 0.3

    # --- Model input dataframe ---
    df = pd.DataFrame({
        'kills': [kills],
        'damageDealt': [damageDealt],
        'walkDistance': [walkDistance],
        'rideDistance': [rideDistance],
        'weaponsAcquired': [weaponsAcquired],
        'totalDistance': [totalDistance],
        'walkRideRatio': [walkRideRatio],
        'damagePerKill': [damagePerKill],
        'playersInGroup': [playersInGroup],
        'playersInMatch': [playersInMatch],
        'combatScore': [combatScore],
        'playerAggression': [playerAggression],
        'matchType': [matchType]
    })

    pred = float(model.predict(df)[0])

            # Tahmini sıra aralığı hesapla
    tahmini_sira = int((1 - pred) * playersInMatch)
    alt_sira = max(1, tahmini_sira - 5)
    ust_sira = min(playersInMatch, tahmini_sira + 5)

    # --- Performans değerlendirmesi yeni kurallara göre ---
    if alt_sira <= 10:
        yorum = "⭐ Üst düzey performans! Oyuncunun maçı ilk 10'da bitirme ihtimali çok yüksek."
        renk = "green"
    elif alt_sira <= 25:
        yorum = "🔥 Yüksek performans! Oyuncu ilk 25 içinde güçlü bir oyun sergiliyor."
        renk = "limegreen"
    elif alt_sira <= 40:
        yorum = "👍 Orta seviye performans. Oyuncu istikrarlı ama daha üst sıralar için gelişebilir."
        renk = "orange"
    elif alt_sira <= 55:
        yorum = "⚠️ Düşük performans. Oyuncu orta-alt sıralarda bitirebilir."
        renk = "darkorange"
    else:
        yorum = "❌ Zayıf performans. Oyuncu erken safhalarda elenme riski taşıyor."
        renk = "red"

    # HTML açıklama
    explanation = f"""
        <div style='font-size:22px; line-height:1.6; padding:12px;'>
            <b>Tahmini WinPlacePerc:</b> {pred:.2f}<br><br>
            <b>Tahmini Bitiş Sırası:</b><br>
            {alt_sira}. - {ust_sira}. sıra aralığı<br><br>
            <b>Maç Türü:</b> {matchTypeName} (kod: {matchType})<br><br>
            <b>Performans Analizi:</b><br>
            <span style='color:{renk}; font-weight:bold;'>{yorum}</span><br><br>
            <i>Not: Sıralama, maçtaki toplam oyuncu sayısına göre hesaplanır.</i>
        </div>
    """



    return pred, explanation


# --- Arayüz inputları ---
inputs = [
    gr.Number(label="Öldürme Sayısı (Kills, max 99)", value=0),
    gr.Number(label="Verilen Hasar (Damage Dealt)", value=0),
    gr.Number(label="Yürüme Mesafesi (Walk Distance)", value=0),
    gr.Number(label="Araç Mesafesi (Ride Distance)", value=0),
    gr.Number(label="Alınan Silah Sayısı (Weapons Acquired)", value=0),

    gr.Dropdown(
        label="Gruptaki Oyuncu Sayısı (1 = Solo, 2 = Duo, 3-4 = Squad)",
        choices=[1, 2, 3, 4],
        value=1
    ),

    gr.Number(label="Maçtaki Oyuncu Sayısı (max 100)", value=100),
]

# --- HTML destekli output ---
outputs = [
    gr.Number(label="Tahmini WinPlacePerc"),
    gr.HTML(label="Açıklama")
]

iface = gr.Interface(
    fn=predict_winplace,
    inputs=inputs,
    outputs=outputs,
    title="PUBG WinPlacePerc Tahmin Aracı",
    description="Oyuncu istatistiklerine göre bitiş yüzdesini tahmin eder."
)

iface.launch()