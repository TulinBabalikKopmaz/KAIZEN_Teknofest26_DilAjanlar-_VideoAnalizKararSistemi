"""Elle video etiketleme arayüzü. Ollama / Qwen gerekmez.

Çalıştırma (proje klasöründen):
    streamlit run app/review_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_frames import safe_id

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "data" / "videos"
LABEL_ROOT = ROOT / "data" / "labels"
PRED_ROOT = ROOT / "data" / "predictions"
FRAMES_ROOT = ROOT / "data" / "frames"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

CATEGORIES = {
    "normal": "Normal — kaza yok, ciddi risk yok",
    "near_miss": "Near miss — kaza olmadı ama neredeyse olacaktı",
    "accident": "Kaza — fiili çarpışma, düşme, devrilme, yaralanma",
}
RISKS = ["Düşük", "Orta", "Yüksek"]
# Belirsizlik dördüncü bir sonuç sınıfı değil; sahnenin okunabilirliği.
# "Kaza var ama görüntüden anlaşılmıyor" → category=accident, ambiguity=belirsiz.
AMBIGUITY = {
    "net": "Net — sahnede ne olduğu tartışmasız",
    "belirsiz": "Belirsiz — makul bir insan farklı yorumlayabilir",
}
AMBIGUITY_REASONS = {
    "": "(gerekçe yok)",
    "gorus_engeli": "Görüş engeli / kötü açı — olay kısmen kapalı",
    "cerceve_disi": "Olay kısmen kare dışında",
    "niyet_belirsiz": "Rutin mi kaza mı belli değil (ör. yerdeki kişi bakım mı yapıyor?)",
    "alarm_gorunumlu_normal": "Normal ama alarm verdirebilir (kaynak kıvılcımı, duman, buhar)",
    "kalite": "Görüntü kalitesi düşük (bulanık, karanlık, düşük fps)",
    "sonuc_gorunmuyor": "Olayın sonucu videoda görünmüyor (kesme / kamera dönüyor)",
    "diger": "Diğer — notta açıkla",
}
EVENT_TYPES = {
    "normal": "Normal akış",
    "near_miss": "Near miss",
    "kaza": "Kaza",
    "risk": "Riskli durum",
    "mudahale": "Müdahale / toplanma",
    "diger": "Diğer",
}
SEVERITIES = {
    "dusuk": "Düşük",
    "orta": "Orta",
    "yuksek": "Yüksek",
}
def list_videos() -> list[Path]:
    return sorted(p for p in VIDEO_ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def empty_label(video: Path) -> dict:
    folder = video.parent.name
    return {
        "video_id": safe_id(video),
        "filename": video.name,
        "category": folder if folder in CATEGORIES else "normal",
        "ambiguity": "net",
        "ambiguity_reason": "",
        "duration_sec": None,
        "status": "in_review",
        "labeled_by": "human",
        "summary": "",
        "events": [],
        "risk": "Orta",
        "actions": [],
        "notes": "",
    }


def load_label(video: Path) -> dict:
    path = LABEL_ROOT / f"{safe_id(video)}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        base = empty_label(video)
        base.update(data)
        return base
    return empty_label(video)


def competition_view(label: dict) -> dict:
    return {
        "summary": label.get("summary", ""),
        "events": [
            {"time": e.get("time", "00:00"), "event": e.get("event", "")}
            for e in label.get("events", [])
            if e.get("event")
        ],
        "risk": label.get("risk", "Orta"),
        "actions": [a for a in label.get("actions", []) if a],
    }


def load_pred(video: Path) -> dict | None:
    path = PRED_ROOT / f"{safe_id(video)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def spec_card(title: str, data: dict | None) -> None:
    st.subheader(title)
    if not data:
        st.info("Bu video için Qwen tahmini yok.")
        return
    st.markdown(f"**Risk:** {data.get('risk', '-')}")
    st.markdown(f"**Özet:** {data.get('summary') or '-'}")
    events = data.get("events") or []
    if events:
        st.markdown("**Olaylar:**")
        for event in events:
            st.markdown(f"- `{event.get('time', '00:00')}` — {event.get('event', '')}")
    else:
        st.markdown("**Olaylar:** (yok)")
    actions = [a for a in (data.get("actions") or []) if a]
    if actions:
        st.markdown("**Aksiyonlar:**")
        for action in actions:
            st.markdown(f"- {action}")
    st.code(json.dumps(competition_view(data), ensure_ascii=False, indent=2), language="json")


def save_label(label: dict) -> Path:
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    dest = LABEL_ROOT / f"{label['video_id']}.json"
    dest.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    spec = LABEL_ROOT / f"{label['video_id']}_spec.json"
    spec.write_text(
        json.dumps(competition_view(label), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest


st.set_page_config(page_title="TEKNOFEST Video Etiketleme", layout="wide")

mode = st.sidebar.radio(
    "Ekran",
    ["qwen", "gold"],
    format_func=lambda x: {
        "qwen": "Qwen çıktısını izle",
        "gold": "Gold etiketleme",
    }[x],
    index=0,
)

st.title("Qwen vs sizin etiketiniz" if mode == "qwen" else "Video etiketleme")
st.caption(
    "Soldan videoyu seçin, oynatın. Solda sizin gold’unuz, sağda Qwen’in yazdığı çıktı."
    if mode == "qwen"
    else "Amacımız: her videoyu izleyip “bu sahnede ne oldu?”yu Türkçe ve zamanlı yazmak."
)

if mode == "gold":
    with st.expander("Bu ekranda ne yapıyorum? (ilk kez okuyun)", expanded=False):
        st.markdown(
            """
1. Soldan bir video seçin, oynatın.
2. Videoda **sadece önemli anları** yazın. Her saniyeyi yazmayın.
   Örnek: `00:15` — Forklift devrildi.
3. Kısa bir özet, risk seviyesi ve operatöre 1–3 aksiyon yazın.
4. Emin değilseniz **Taslak** kaydedin. İzleyip doğru olduğuna inandıysanız **Gold** kaydedin.
5. **Gold = cevap anahtarı.** Sonra Qwen/ajan aynı videoyu analiz edince
   onun çıktısını sizin gold etiketinizle karşılaştıracağız.

Ollama kurulu olması gerekmez. Bu adım tamamen sizin izleyip yazmanız.
            """
        )

videos = list_videos()
if not videos:
    st.error("Henüz etiketlenecek video yok.")
    st.markdown(
        """
Videolarınızı Finder ile şu üç klasöre kopyalayın, sonra bu sayfayı yenileyin:

- `data/videos/normal/` — kaza olmayanlar
- `data/videos/near_miss/` — neredeyse kaza
- `data/videos/accident/` — kaza olanlar
        """
    )
    st.code(str(VIDEO_ROOT), language="text")
    st.stop()

def video_status(path: Path) -> str:
    p = LABEL_ROOT / f"{safe_id(path)}.json"
    if not p.exists():
        return "yok"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") or "yok"
    except json.JSONDecodeError:
        return "yok"


st.sidebar.markdown("### Filtre")
cat_filter = st.sidebar.selectbox(
    "Kategori",
    ["hepsi", "accident", "near_miss", "normal"],
    format_func=lambda x: {
        "hepsi": "Hepsi",
        "accident": "Kaza",
        "near_miss": "Near miss",
        "normal": "Normal",
    }[x],
)
only_gold = False
only_drafts = False
only_pending_ambiguity = False
if mode == "gold":
    only_gold = st.sidebar.checkbox("Sadece gold (birleşik cevap anahtarı)", value=True)
    only_drafts = st.sidebar.checkbox(
        "Sadece model taslağı bekleyenler",
        value=False,
        help="status=auto: model yazdı, insan onayı bekliyor. Onay kuyruğunu buradan eritin.",
    )
    only_pending_ambiguity = st.sidebar.checkbox(
        "Sadece belirsizlik onayı bekleyenler",
        value=False,
        help="Otomatik önerilmiş ya da hiç işaretlenmemiş videolar; hızlı gözden geçirme turu için.",
    )


def label_field(path: Path, field: str) -> str:
    p = LABEL_ROOT / f"{safe_id(path)}.json"
    if not p.exists():
        return ""
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get(field) or "")
    except json.JSONDecodeError:
        return ""


filtered = videos
if cat_filter != "hepsi":
    filtered = [v for v in filtered if v.parent.name == cat_filter]
if only_gold:
    filtered = [v for v in filtered if video_status(v) == "gold"]
if only_drafts:
    filtered = [v for v in filtered if video_status(v) == "auto"]
if only_pending_ambiguity:
    filtered = [v for v in filtered if label_field(v, "ambiguity_source") != "human"]
if mode == "qwen":
    filtered = [v for v in filtered if load_pred(v)]

if not filtered:
    if mode == "qwen":
        st.warning(
            "Henüz Qwen tahmini yok. Soldaki kategoriyi “Hepsi” yapın "
            "veya `scripts/run_kpi_sample.py` ile örnek videoları tekrar çalıştırın."
        )
    else:
        st.warning("Bu filtreye uyan video yok. Soldaki filtreyi genişletin.")
    st.stop()

names = [f"{p.parent.name} / {p.name}" for p in filtered]
choice = st.sidebar.selectbox("Hangi videoyu inceliyorum?", names)
video = filtered[names.index(choice)]
label = load_label(video)

saved = sum(1 for v in videos if (LABEL_ROOT / f"{safe_id(v)}.json").exists())
gold = sum(1 for v in videos if video_status(v) == "gold")

amb_done = sum(1 for v in videos if label_field(v, "ambiguity_source") == "human")
amb_marked = sum(1 for v in videos if label_field(v, "ambiguity") == "belirsiz")

st.sidebar.markdown("### İlerleme")
st.sidebar.metric("Listede görünen", f"{len(filtered)}")
st.sidebar.metric("Gold (onaylı cevap anahtarı)", f"{gold} / {len(videos)}")
drafts = sum(1 for v in videos if video_status(v) == "auto")
st.sidebar.metric(
    "Belirsizlik onaylandı",
    f"{amb_done} / {len(videos)}",
    help=f"Belirsiz işaretli: {amb_marked} — şartname hedefi videoların ~yüzde 20'si",
)
st.sidebar.metric("Onay bekleyen model taslağı", f"{drafts}")
st.sidebar.caption("Kılavuz: docs/etiketleme_kilavuzu.md")
st.sidebar.info("Birleşik gold: sizin kazalarınız + Mustafa'nın normal/near miss kayıtları.")
if mode == "qwen":
    st.sidebar.caption("Şu an 6 örnek videoda Qwen çıktısı var. Videoyu oynatıp altta karşılaştırın.")

if mode == "qwen":
    pred = load_pred(video)
    st.video(str(video))
    st.write(f"Dosya klasörü: **{video.parent.name}** — `{video.name}`")
    gold_col, qwen_col = st.columns(2)
    with gold_col:
        spec_card("Sizin gold etiketiniz (cevap anahtarı)", label if label.get("status") == "gold" else label)
    with qwen_col:
        spec_card("Qwen’in tahmini (öğrenci)", pred)
    st.stop()

left, right = st.columns([1.15, 1])

with left:
    st.subheader("1. Videoyu izle")
    if label.get("status") == "auto":
        st.info(
            f"Bu videoda **model taslağı** var ({label.get('labeled_by', 'model')}). "
            "Aşağıdaki alanlar taslakla dolu geldi: sıfırdan yazmak yerine kontrol edip "
            "düzeltin, sonra Gold onayı verin.",
            icon="✏️",
        )
    st.video(str(video))
    st.write(f"Dosya klasörü: **{video.parent.name}** — `{video.name}`")
    st.caption("Oynatıcıdaki süre, olay satırına yazacağınız 00:15 gibi zamanın kaynağıdır.")

    strip = sorted((FRAMES_ROOT / video.parent.name / safe_id(video)).glob("*.jpg"))
    if strip:
        with st.expander(f"Zaman damgalı kareler ({len(strip)}) — olay saniyesini hızlı bulmak için"):
            for row_start in range(0, len(strip), 4):
                cols = st.columns(4)
                for col, frame_path in zip(cols, strip[row_start : row_start + 4]):
                    col.image(str(frame_path), caption=frame_path.stem.replace("-", ":"))

with right:
    st.subheader("2. Videoyu sınıflandır")
    category_keys = list(CATEGORIES)
    category = st.selectbox(
        "Bu video genel olarak ne?",
        category_keys,
        index=category_keys.index(label["category"]) if label["category"] in category_keys else 0,
        format_func=lambda k: CATEGORIES[k],
    )
    risk = st.selectbox(
        "Operatör için risk seviyesi",
        RISKS,
        index=RISKS.index(label["risk"]) if label["risk"] in RISKS else 1,
        help="Normal sahnede genelde Düşük, near miss Orta, kaza Yüksek.",
    )
    amb_keys = list(AMBIGUITY)
    current_amb = label.get("ambiguity") or "net"
    ambiguity = st.radio(
        "Sahne ne kadar okunabilir?",
        amb_keys,
        index=amb_keys.index(current_amb) if current_amb in amb_keys else 0,
        format_func=lambda k: AMBIGUITY[k],
        horizontal=True,
        help=(
            "Kategoriyi değiştirmez. Olayın gerçek sonucunu yukarıda yazın; "
            "burada sadece 'görüntüden anlaşılıyor mu' sorusunu cevaplıyorsunuz. "
            "Şartname test kümesinin %20'si belirsiz olmalı."
        ),
    )
    reason_keys = list(AMBIGUITY_REASONS)
    current_reason = label.get("ambiguity_reason") or ""
    ambiguity_reason = ""
    if ambiguity == "belirsiz":
        ambiguity_reason = st.selectbox(
            "Neden belirsiz?",
            reason_keys,
            index=reason_keys.index(current_reason) if current_reason in reason_keys else 0,
            format_func=lambda k: AMBIGUITY_REASONS[k],
        )
    if label.get("ambiguity_source") == "auto":
        st.caption("Bu belirsizlik işareti otomatik önerildi, sizin onayınızı bekliyor.")
    summary = st.text_area(
        "Kısa Türkçe özet (1–2 cümle)",
        value=label.get("summary", ""),
        height=110,
        placeholder="Örnek: Forklift yük taşırken dengesi bozuldu ve devrildi. Yerde hareketsiz bir kişi var.",
    )
    actions_text = st.text_area(
        "Operatöre aksiyonlar (her satır bir öneri)",
        value="\n".join(label.get("actions", [])),
        height=100,
        placeholder="Sağlık ekibini çağır\nAlanı güvenlik altına al",
    )
    notes = st.text_area(
        "Takım notu (şartnameye gitmez)",
        value=label.get("notes", ""),
        height=70,
        placeholder="Görüntü bulanık, 00:12'de ne olduğu tam belli değil...",
    )

st.subheader("3. Kritik anları yaz")
st.markdown(
    "Her saniyeyi değil, **olayın değiştiği anı** yazın. Zamanı `MM:SS` yazın (dakika:saniye). "
    "Yeni satır için tablonun altındaki **+** işaretine basın."
)

raw_events = label.get("events") or []
if not raw_events:
    raw_events = [
        {
            "time": "00:00",
            "time_end": "",
            "event": "",
            "event_type": "diger",
            "severity": "orta",
        }
    ]

edited = st.data_editor(
    raw_events,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "time": st.column_config.TextColumn("Zaman (00:15)", required=True, width="small"),
        "time_end": st.column_config.TextColumn("Bitiş (opsiyonel)", width="small"),
        "event": st.column_config.TextColumn("Ne oldu? (Türkçe cümle)", required=True),
        "event_type": st.column_config.SelectboxColumn(
            "Olay türü", options=list(EVENT_TYPES), width="small"
        ),
        "severity": st.column_config.SelectboxColumn(
            "Şiddet", options=list(SEVERITIES), width="small"
        ),
    },
)

st.subheader("4. Şartnameye gidecek JSON (önizleme)")
preview = {
    "summary": summary.strip(),
    "events": [
        {"time": e.get("time", "00:00"), "event": e.get("event", "").strip()}
        for e in edited
        if str(e.get("event", "")).strip()
    ],
    "risk": risk,
    "actions": [line.strip() for line in actions_text.splitlines() if line.strip()],
}
st.code(json.dumps(preview, ensure_ascii=False, indent=2), language="json")

missing = []
if not preview["summary"]:
    missing.append("özet")
if not preview["events"]:
    missing.append("en az bir olay")
if not preview["actions"]:
    missing.append("en az bir aksiyon")
if missing:
    st.warning("Gold kaydetmeden önce doldurun: " + ", ".join(missing))

st.subheader("5. Kaydet")
st.markdown(
    "**Taslak:** ara kayıt, sonra döneceğim.  "
    "**Gold:** videoyu izledim, yazdıklarım doğru, cevap anahtarı olsun."
)

c1, c2 = st.columns(2)
payload_base = {
    "video_id": safe_id(video),
    "filename": video.name,
    "category": category,
    "ambiguity": ambiguity,
    "ambiguity_reason": ambiguity_reason,
    "ambiguity_source": "human",
    "duration_sec": label.get("duration_sec"),
    "labeled_by": "human",
    "summary": preview["summary"],
    "events": [e for e in edited if str(e.get("event", "")).strip()],
    "risk": risk,
    "actions": preview["actions"],
    "notes": notes,
}

with c1:
    if st.button("Taslak kaydet", use_container_width=True):
        dest = save_label({**payload_base, "status": "in_review"})
        st.success(f"Taslak kaydedildi: `{dest.name}`")
        st.rerun()

with c2:
    gold_ok = not missing
    if st.button("Gold olarak onayla", type="primary", use_container_width=True, disabled=not gold_ok):
        dest = save_label({**payload_base, "status": "gold"})
        st.success(f"Gold kaydedildi. Bu video artık cevap anahtarı: `{dest.name}`")
        st.rerun()
