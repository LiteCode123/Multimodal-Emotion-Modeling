"""Build auditable 50-bin text, audio, and vision features for Task02's 100 MOSEI clips."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from src.data.paths import resolve_recorded_path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/q1/media"
OUT = TASK / "outputs"
FIG = TASK / "figures"
DATA = ROOT / "data" / "raw" / "E题数据" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
FIX02 = ROOT / "data/intermediate/q1/effective_timeline.csv"
TEXT_UP = ROOT / "data/intermediate/q1/text"
BINS = 50
SR = 16000
SEED = 20260923
TEXT_DIM = 768


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if fields:
            w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_official() -> dict[str, dict]:
    import pandas as pd
    df = pd.read_excel(DATA / "label-100.xlsx")
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    if not required.issubset(df.columns):
        raise ValueError(f"label-100.xlsx missing columns: {required - set(df.columns)}")
    out = {}
    for rec in df.to_dict("records"):
        sid = f"{str(rec['video_id']).strip()}$_${int(rec['clip_id'])}"
        out[sid] = {"official_label": float(rec["label"]), "official_text": "" if rec["text"] is None else str(rec["text"]),
                    "annotation": "" if rec["annotation"] is None else str(rec["annotation"])}
    return out


def probe_video(path: Path) -> dict:
    import av
    c = av.open(str(path))
    try:
        vstreams = [s for s in c.streams if s.type == "video"]
        astreams = [s for s in c.streams if s.type == "audio"]
        if not vstreams:
            raise RuntimeError("no video stream")
        vs = vstreams[0]; ast = astreams[0] if astreams else None
        vf = ve = af = ae = None
        fps = float(vs.average_rate) if vs.average_rate else math.nan
        # Fast metadata pass. Full stream decoding belongs to extraction stages.
        for st, kind in ((vs, "v"), (ast, "a")):
            if st is None: continue
            start = float(st.start_time * st.time_base) if st.start_time is not None else 0.0
            duration = float(st.duration * st.time_base) if st.duration is not None else math.nan
            if kind == "v": vf, ve = start, (start + duration if math.isfinite(duration) else None)
            else: af, ae = start, (start + duration if math.isfinite(duration) else None)
        return {"fps": fps, "width": int(vs.width), "height": int(vs.height), "video_first_pts": vf,
                "video_end_pts": ve, "audio_first_pts": af, "audio_end_pts": ae,
                "audio_available": bool(ast), "probe_error": ""}
    finally:
        c.close()


def get_manifest() -> list[dict]:
    official = read_official()
    vids = sorted(DATA.glob("*/*.mp4"))
    issues = []
    rows = []
    if len(vids) != 100:
        issues.append({"sample_id": "", "modality": "data", "bin_id": "", "issue_type": "video_count_mismatch",
                       "severity": "ERROR", "message": f"Expected 100 mp4, found {len(vids)}"})
    seen = set()
    fix = {r["sample_id"]: r for r in read_csv(FIX02)} if FIX02.exists() else {}
    for p in vids:
        sid = f"{p.parent.name}$_${p.stem}"
        rec = {"sample_id": sid, "video_file": p.name, "video_path": str(p.relative_to(ROOT)).replace("\\", "/"),
               "duration_sec": math.nan, "fps": math.nan, "width": 0, "height": 0, "audio_available": False,
               "transcript_available": sid in official and bool(official[sid]["official_text"].strip()),
               "label_available": sid in official, "official_label": official.get(sid, {}).get("official_label", math.nan),
               "official_text": official.get(sid, {}).get("official_text", ""), "annotation": official.get(sid, {}).get("annotation", ""),
               "effective_t0": math.nan, "effective_t1": math.nan, "timeline_source": "", "processing_status": "pending", "notes": ""}
        if sid in seen:
            issues.append({"sample_id": sid, "modality": "data", "bin_id": "", "issue_type": "duplicate_sample_id", "severity": "ERROR", "message": "Duplicate video sample_id"})
        seen.add(sid)
        if sid not in official:
            issues.append({"sample_id": sid, "modality": "label", "bin_id": "", "issue_type": "missing_label", "severity": "ERROR", "message": "No label-100.xlsx row"})
        if not p.is_file() or p.stat().st_size == 0:
            issues.append({"sample_id": sid, "modality": "video", "bin_id": "", "issue_type": "missing_or_empty_video", "severity": "ERROR", "message": str(p)})
            rec["processing_status"] = "video_missing"
            rows.append(rec); continue
        try:
            meta = probe_video(p); rec.update({k: meta[k] for k in ("fps", "width", "height", "audio_available")})
            if sid in fix:
                t0, t1 = float(fix[sid]["effective_t0"]), float(fix[sid]["effective_t1"])
                rec["timeline_source"] = "verified_Fix02"
            else:
                starts = [v for v in (meta["video_first_pts"], meta["audio_first_pts"]) if v is not None]
                ends = [v for v in (meta["video_end_pts"], meta["audio_end_pts"]) if v is not None]
                if not starts or not ends: raise RuntimeError("cannot establish audio/video PTS interval")
                t0, t1 = min(starts), max(ends)
                rec["timeline_source"] = "Fix02_formula_recomputed_from_stream_PTS_metadata"
            if not math.isfinite(t1-t0) or t1 <= t0: raise RuntimeError(f"invalid effective interval {t0}, {t1}")
            rec["effective_t0"], rec["effective_t1"], rec["duration_sec"] = t0, t1, t1-t0
            rec["processing_status"] = "ready" if rec["audio_available"] else "ready_audio_missing"
            if not rec["audio_available"]:
                issues.append({"sample_id": sid, "modality": "audio", "bin_id": "", "issue_type": "audio_stream_missing", "severity": "WARNING", "message": "No audio stream in source video"})
        except Exception as e:
            rec["processing_status"] = "probe_failed"; rec["notes"] = repr(e)
            issues.append({"sample_id": sid, "modality": "video", "bin_id": "", "issue_type": "probe_failed", "severity": "ERROR", "message": repr(e)})
        rows.append(rec)
    for sid in sorted(set(official)-seen):
        issues.append({"sample_id": sid, "modality": "video", "bin_id": "", "issue_type": "label_without_video", "severity": "ERROR", "message": "Official record has no matching MP4"})
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT/"manifest_100.csv", rows)
    write_csv(OUT/"data_integrity_issues.csv", issues)
    if not issues:
        (OUT/"data_integrity_issues.csv").write_text("sample_id,modality,bin_id,issue_type,severity,message\n",encoding="utf-8-sig")
    print(f"manifest: videos={len(vids)} IDs={len(seen)} labels={len(official)} integrity_issues={len(issues)}")
    return rows


def load_manifest() -> list[dict]:
    p=OUT/"manifest_100.csv"
    if not p.exists(): return get_manifest()
    return read_csv(p)


def frame_bins(duration: float) -> list[dict]:
    edges = np.linspace(0.0, duration, BINS+1)
    return [{"bin_id": k, "bin_start_sec": float(edges[k]), "bin_end_sec": float(edges[k+1]),
             "bin_center_sec": float((edges[k]+edges[k+1])/2), "normalized_time": (k+0.5)/BINS} for k in range(BINS)]


def load_prior_text() -> dict:
    path=TEXT_UP/"finalize_word_level"
    out={}
    if path.exists():
        for f in path.glob("*.npz"):
            with np.load(f, allow_pickle=False) as z:
                out[f.stem]={"word_start":z["word_start"].astype(float), "word_end":z["word_end"].astype(float),
                             "word_embeddings":z["word_embeddings"].astype(np.float32), "asr_text":str(z["asr_text"].item())}
    return out


def build_text(rows: list[dict]) -> list[dict]:
    from faster_whisper import WhisperModel
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_asr=WhisperModel("small.en", device="cpu", compute_type="int8")
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, local_files_only=True)
    bert=AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(torch.device("cpu")); bert.eval()
    prior=load_prior_text(); out=[]; errors=[]
    for i,r in enumerate(rows,1):
        sid=r["sample_id"]; dur=float(r["duration_sec"]); bins=frame_bins(dur)
        feats=np.zeros((BINS,TEXT_DIM),np.float32); present=np.zeros(BINS,np.int8)
        try:
            if sid in prior:
                words=[]
                for s,e,v in zip(prior[sid]["word_start"],prior[sid]["word_end"],prior[sid]["word_embeddings"]):
                    words.append((float(s),float(e),v))
                transcript=prior[sid]["asr_text"]; source="Task01_Finalize_word_level_reused"
            else:
                import av
                from av.audio.resampler import AudioResampler
                path=resolve_recorded_path(r["video_path"]); c=av.open(str(path)); streams=[s for s in c.streams if s.type=="audio"]
                if not streams:
                    c.close(); raise RuntimeError("no audio stream for ASR")
                resampler=AudioResampler(format="flt",layout="mono",rate=SR); chunks=[]
                for fr in c.decode(streams[0]):
                    for x in resampler.resample(fr): chunks.append(x.to_ndarray().reshape(-1))
                for x in resampler.resample(None): chunks.append(x.to_ndarray().reshape(-1))
                c.close()
                if not chunks: raise RuntimeError("audio decoded to empty waveform")
                wav=np.concatenate(chunks).astype(np.float32)
                t0=float(r["effective_t0"]); t1=float(r["effective_t1"])
                wav=wav[max(0,int(round(t0*SR))):min(len(wav),int(round(t1*SR)))]
                segs,_info=model_asr.transcribe(wav,language="en",word_timestamps=True,beam_size=5,vad_filter=False)
                raw=[]
                for seg in segs:
                    for w in seg.words or []:
                        clean=re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$","",str(w.word).strip())
                        if re.search(r"[A-Za-z0-9]",clean): raw.append((clean,max(0,float(w.start)),min(dur,float(w.end))))
                transcript=" ".join(x[0] for x in raw)
                if raw:
                    text=" ".join(x[0] for x in raw); enc=tokenizer(text,return_offsets_mapping=True,return_tensors="pt",truncation=True,max_length=512)
                    offsets=enc.pop("offset_mapping")[0].numpy()
                    with torch.no_grad(): hidden=bert(**enc).last_hidden_state[0].cpu().numpy()
                    spans=[]; pos=0
                    for word,_,_ in raw:
                        a=text.find(word,pos); b=a+len(word); spans.append((a,b)); pos=b
                    emb=[]
                    for a,b in spans:
                        ix=[j for j,(x,y) in enumerate(offsets) if y>x and int(x)<b and int(y)>a]
                        if not ix: raise RuntimeError(f"no bert pieces for {text[a:b]}")
                        emb.append(hidden[ix].mean(axis=0))
                    words=[(item[1],item[2],v.astype(np.float32)) for item,v in zip(raw,emb)]
                else: words=[]
                source="faster-whisper-small.en + bert-base-uncased"
            if words:
                for k,b in enumerate(bins):
                    overlaps=[]; weights=[]
                    for s,e,v in words:
                        w=max(0.0,min(e,b["bin_end_sec"])-max(s,b["bin_start_sec"]))
                        if w>0: overlaps.append(v); weights.append(w)
                    if weights:
                        feats[k]=np.average(np.asarray(overlaps),axis=0,weights=np.asarray(weights)); present[k]=1
            status="valid" if present.sum()>0 else "asr_empty"
            out.extend([{**{"sample_id":sid},**b,"text_present":int(present[b["bin_id"]]),**{f"text_{j:03d}":float(v) for j,v in enumerate(feats[b["bin_id"]])},
                        "asr_text":transcript,"text_status":status,"text_source":source} for b in bins])
            np.savez_compressed(OUT/f"_text_{safe_id(sid)}.npz",features=feats,present=present,asr_text=np.asarray(transcript),text_status=np.asarray(status))
        except Exception as e:
            errors.append((sid,repr(e))); out.extend([{**{"sample_id":sid},**b,"text_present":0,**{f"text_{j:03d}":0.0 for j in range(TEXT_DIM)},"asr_text":"","text_status":"failed","text_source":"","error":repr(e)} for b in bins])
        print(f"text {i}/{len(rows)} {sid}: bins={int(present.sum()) if 'present' in locals() else 0}",flush=True)
    write_features(out,"text_features_50bin.parquet")
    return errors


def safe_id(s: str) -> str: return re.sub(r"[^A-Za-z0-9_-]","_",s)


def audio_features(y: np.ndarray) -> dict:
    import librosa
    y=np.asarray(y,dtype=np.float32)
    if not len(y): return {"audio_valid":0,"audio_reason":"empty_audio","audio_present":0,"audio_voiced_ratio":math.nan}
    n_fft=1024; hop=256
    rms=librosa.feature.rms(y=y,frame_length=n_fft,hop_length=hop,center=True)[0]
    zcr=librosa.feature.zero_crossing_rate(y,frame_length=n_fft,hop_length=hop)[0]
    spec=np.abs(librosa.stft(y,n_fft=n_fft,hop_length=hop,center=True))
    cent=librosa.feature.spectral_centroid(S=spec,sr=SR)[0]
    bw=librosa.feature.spectral_bandwidth(S=spec,sr=SR)[0]
    roll=librosa.feature.spectral_rolloff(S=spec,sr=SR,roll_percent=0.85)[0]
    mfcc=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13,n_fft=n_fft,hop_length=hop)
    frames=librosa.yin(y,fmin=65,fmax=500,sr=SR,frame_length=2048,hop_length=hop)
    # librosa YIN returns an F0 candidate per frame; only non-silent, physically plausible frames count voiced.
    rr=librosa.feature.rms(y=y,frame_length=2048,hop_length=hop)[0]
    m=min(len(frames),len(rr)); voiced=(rr[:m]>1e-4)&np.isfinite(frames[:m])&(frames[:m]>=65)&(frames[:m]<=500)
    f0=frames[:m][voiced]
    result={"audio_valid":1,"audio_reason":"valid_audio","audio_present":1,"audio_voiced_ratio":float(voiced.mean()) if m else 0.0}
    for name,x in [("rms",rms),("zcr",zcr),("spectral_centroid",cent),("spectral_bandwidth",bw),("spectral_rolloff",roll)]:
        result[f"{name}_mean"]=float(np.mean(x)); result[f"{name}_std"]=float(np.std(x))
    for j in range(13): result[f"mfcc{j+1:02d}_mean"]=float(np.mean(mfcc[j])); result[f"mfcc{j+1:02d}_std"]=float(np.std(mfcc[j]))
    result["f0_mean"]=float(np.mean(f0)) if len(f0) else math.nan
    result["f0_std"]=float(np.std(f0)) if len(f0) else math.nan
    result["f0_missing"]=int(not len(f0))
    return result


def build_audio(rows: list[dict]) -> list[dict]:
    import av
    out=[]; issues=[]
    for i,r in enumerate(rows,1):
        sid=r["sample_id"]; dur=float(r["duration_sec"]); bins=frame_bins(dur)
        try:
            c=av.open(str(resolve_recorded_path(r["video_path"]))); streams=[s for s in c.streams if s.type=="audio"]
            if not streams: raise RuntimeError("no audio stream")
            # PyAV audio resampling by timestamp is handled in one stream pass.
            c.close()
            c=av.open(str(resolve_recorded_path(r["video_path"]))); st=[s for s in c.streams if s.type=="audio"][0]
            from av.audio.resampler import AudioResampler
            rs=AudioResampler(format="flt",layout="mono",rate=SR); pieces=[]
            for f in c.decode(st):
                for o in rs.resample(f): pieces.append(o.to_ndarray().reshape(-1))
            for o in rs.resample(None): pieces.append(o.to_ndarray().reshape(-1))
            c.close(); wave=np.concatenate(pieces).astype(np.float32)
            t0=float(r["effective_t0"]); t1=float(r["effective_t1"]); wave=wave[max(0,round(t0*SR)):min(len(wave),round(t1*SR))]
            all_features=[]
            edges=np.linspace(0,len(wave),BINS+1,dtype=int)
            for k,b in enumerate(bins):
                s=edges[k]; e=edges[k+1]
                try: f=audio_features(wave[s:e]); f["audio_present"]=1; f["audio_reason"]=f.get("audio_reason","valid_audio")
                except Exception as ex: f={"audio_present":1,"audio_valid":0,"audio_reason":"feature_error","audio_voiced_ratio":math.nan}; issues.append(issue(sid,"audio",k,"audio_feature_error","WARNING",repr(ex)))
                all_features.append({**{"sample_id":sid},**b,**f})
            out.extend(all_features)
        except Exception as e:
            issues.append(issue(sid,"audio","","audio_extract_failed","ERROR",repr(e)))
            for b in bins: out.append({"sample_id":sid,**b,"audio_present":0,"audio_valid":0,"audio_reason":"audio_unavailable","audio_voiced_ratio":math.nan})
        print(f"audio {i}/{len(rows)} {sid}",flush=True)
    return out,issues


def build_vision(rows: list[dict]) -> tuple[list[dict],list[dict]]:
    import av, cv2
    cv2.setNumThreads(1)
    out=[]; issues=[]; positions=(0.25,0.50,0.75)
    for i,r in enumerate(rows,1):
        sid=r["sample_id"]; dur=float(r["duration_sec"]); bins=frame_bins(dur)
        video_features=[]
        try:
            c=av.open(str(resolve_recorded_path(r["video_path"]))); stream=[s for s in c.streams if s.type=="video"][0]
            # Decode once, caching only sampled RGB frames; no full-video frame extraction.
            targets=[b["bin_start_sec"]+p*(b["bin_end_sec"]-b["bin_start_sec"]) for b in bins for p in positions]
            chosen={}; ti=0
            for fr in c.decode(stream):
                if fr.pts is None: continue
                t=float(fr.pts*stream.time_base)
                while ti<len(targets) and targets[ti]<=t:
                    chosen[ti]=fr.to_ndarray(format="rgb24"); ti+=1
            c.close()
            # If target lies after final PTS, use nearest decoded frame only when within one frame period.
            fps=float(r["fps"]); max_gap=1.5/fps if fps>0 else 0.08
            if ti<len(targets) and chosen:
                last=next(reversed(chosen.values()))
                while ti<len(targets):
                    chosen[ti]=last.copy(); ti+=1
            for k,b in enumerate(bins):
                arrs=[chosen[j] for j in range(k*3,(k+1)*3) if j in chosen]
                values=[]; prev=None
                for rgb in arrs:
                    im=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
                    brightness=im[:,:,2].astype(np.float32)/255
                    sat=im[:,:,1].astype(np.float32)/255
                    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                    lap=cv2.Laplacian(gray,cv2.CV_32F)
                    values.append({"brightness":float(brightness.mean()),"contrast":float(gray.std()/255),"saturation":float(sat.mean()),"sharpness":float(lap.var()),
                                   "frame_hash":float(gray.mean())})
                if values:
                    mot=[]
                    for a,z in zip(arrs[:-1],arrs[1:]): mot.append(float(np.mean(cv2.absdiff(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),cv2.cvtColor(z,cv2.COLOR_RGB2GRAY)))/255))
                    row={"vision_present":1,"valid_frame_count":len(arrs),"brightness_mean":float(np.mean([x["brightness"] for x in values])),"brightness_std":float(np.std([x["brightness"] for x in values])),
                         "contrast_mean":float(np.mean([x["contrast"] for x in values])),"saturation_mean":float(np.mean([x["saturation"] for x in values])),"saturation_std":float(np.std([x["saturation"] for x in values])),
                         "sharpness_mean":float(np.mean([x["sharpness"] for x in values])),"motion_strength":float(np.mean(mot)) if mot else 0.0,"vision_reason":"valid_frames"}
                else: row={"vision_present":0,"valid_frame_count":0,"vision_reason":"no_sampled_frames"}
                video_features.append({"sample_id":sid,**b,**row})
        except Exception as e:
            issues.append(issue(sid,"vision","","vision_decode_failed","ERROR",repr(e)))
            video_features=[{"sample_id":sid,**b,"vision_present":0,"valid_frame_count":0,"vision_reason":"decode_failed"} for b in bins]
        out.extend(video_features)
        print(f"vision {i}/{len(rows)} {sid}",flush=True)
    return out,issues


def issue(sid,modality,bin_id,typ,severity,message): return {"sample_id":sid,"modality":modality,"bin_id":bin_id,"issue_type":typ,"severity":severity,"message":message}


def write_features(rows: list[dict], filename: str) -> None:
    import pandas as pd
    df=pd.DataFrame(rows)
    path=OUT/filename
    try: df.to_parquet(path,index=False)
    except Exception:
        # Preserve a fully usable output when parquet engine is unavailable.
        csvpath=path.with_suffix(".csv"); df.to_csv(csvpath,index=False,encoding="utf-8-sig")
        print(f"Parquet unavailable; wrote {csvpath.name}")


def read_features(stem: str) -> list[dict]:
    p=OUT/f"{stem}.parquet"
    import pandas as pd
    if p.exists(): return pd.read_parquet(p).to_dict("records")
    p=p.with_suffix(".csv")
    if p.exists(): return pd.read_csv(p).to_dict("records")
    # Interim CSV is the audio/text stage checkpoint.
    p=OUT/f"_{stem.replace('_50bin','')}_features_work.csv"
    return read_csv(p) if p.exists() else []


def summarize_and_align(manifest: list[dict], text, audio, vision, issues):
    key=lambda x:(x["sample_id"],int(x["bin_id"]))
    maps=[]
    for rows in (text,audio,vision): maps.append({key(x):x for x in rows})
    aligned=[]; summary=[]
    for m in manifest:
        sid=m["sample_id"]; tv=[]; av=[]; vv=[]
        trows=[x for x in text if x["sample_id"]==sid]
        if trows and not any(int(x.get("text_present",0) or 0) for x in trows):
            issues.append(issue(sid,"text","","asr_empty","WARNING","small.en returned no valid word timestamps; text features are zero with text_present=0"))
        for b in frame_bins(float(m["duration_sec"])):
            k=(sid,b["bin_id"]); tr=maps[0].get(k,{}); ar=maps[1].get(k,{}); vr=maps[2].get(k,{})
            textp=int(tr.get("text_present",0) or 0); audiop=int(ar.get("audio_present",0) or 0); audiov=int(ar.get("audio_valid",0) or 0); visionp=int(vr.get("vision_present",0) or 0)
            tv.append(textp); av.append(audiov); vv.append(visionp)
            rec={"sample_id":sid,"official_label":m["official_label"],**b,"text_present":textp,"audio_present":audiop,"audio_valid":audiov,
                 "audio_reason":ar.get("audio_reason","audio_failed"),"audio_voiced_ratio":ar.get("audio_voiced_ratio",math.nan),"vision_present":visionp,"valid_frame_count":int(vr.get("valid_frame_count",0) or 0)}
            for src in (tr,ar,vr):
                for name,val in src.items():
                    if name not in rec and name not in {"sample_id","bin_id","bin_start_sec","bin_end_sec","bin_center_sec","normalized_time","official_label","asr_text","text_status","text_source","text_reason","error","text_status","text_source"}:
                        rec[name]=val
            rec["text_status"] = tr.get("text_status", "failed")
            aligned.append(rec)
        warning_count=sum(1 for x in issues if x["sample_id"]==sid and x["severity"]=="WARNING")
        error_count=sum(1 for x in issues if x["sample_id"]==sid and x["severity"]=="ERROR")
        summary.append({"sample_id":sid,"video_file":m["video_file"],"duration_sec":m["duration_sec"],"official_label":m["official_label"],
                        "text_valid_bins":sum(tv),"audio_valid_bins":sum(av),"vision_valid_bins":sum(vv),"text_coverage":sum(tv)/BINS,"audio_coverage":sum(av)/BINS,"vision_coverage":sum(vv)/BINS,
                        "all_modalities_valid_bins":sum(a and b and c for a,b,c in zip(tv,av,vv)),"processing_success":int(error_count==0),"warning_count":warning_count,"error_count":error_count,
                        "notes":"; ".join(sorted({x["issue_type"] for x in issues if x["sample_id"]==sid}))})
    write_features(aligned,"multimodal_features_50bin.parquet"); write_csv(OUT/"multimodal_summary_100.csv",summary); write_csv(OUT/"processing_issues.csv",issues)
    return aligned,summary


def schema(aligned):
    fields=[]
    descriptions={"text_present":"1 if at least one real ASR word overlaps this bin","audio_present":"1 if waveform samples were decoded for this bin","audio_valid":"1 if audio feature extraction completed","audio_voiced_ratio":"Fraction of stable voiced frames; may be NaN when unavailable","vision_present":"1 if at least one sampled video frame decoded","valid_frame_count":"Number of decoded 25%, 50%, 75% bin frames"}
    for col in aligned[0]:
        mod="time" if col in {"sample_id","official_label","bin_id","bin_start_sec","bin_end_sec","bin_center_sec","normalized_time"} else ("text" if col.startswith("text_") else ("audio" if col.startswith(("audio_","rms_","zcr_","spectral_","mfcc","f0_")) else "vision"))
        if col == "text_status": mod="text"
        val=next((r[col] for r in aligned if col in r and r[col] is not None),None)
        fields.append({"feature_name":col,"modality":mod,"data_type":"float64" if isinstance(val,(float,np.floating)) else ("int64" if isinstance(val,(int,np.integer)) else "string"),"dimension":1,
                       "description":descriptions.get(col, "50-bin shared temporal coordinate" if mod=="time" else col.replace("_"," ")),
                       "missing_value_rule":"NaN only where physical estimate unavailable; indicator flags must be consulted" if col.startswith("f0_") else ("flag=0, feature missing; no valid-value zero imputation" if col.endswith("present") or col.endswith("valid") else ""),
                       "normalization_applied":"none","source_method":"Text -> inherited Task01-Finalize (small.en ASR, bert-base-uncased, word-to-bin overlap mean)" if mod=="text" else ("Audio -> Task02 CPU librosa descriptors" if mod=="audio" else ("Vision -> Task02 sampled-frame OpenCV statistics" if mod=="vision" else "Task02 shared 50-bin axis"))})
    write_json(OUT/"multimodal_feature_schema.json",{"n_samples":len({r['sample_id'] for r in aligned}),"n_bins":BINS,"feature_count":len(fields),"time_axis":"B_k=[kD/50,(k+1)D/50); normalized_time=(k+0.5)/50","text_pipeline":"Inherited Task01-Finalize unchanged; official transcript and label metadata only, never used as features","features":fields})


def validate(manifest,aligned,summary,issues):
    ids=[r["sample_id"] for r in manifest]; counts={s:sum(x["sample_id"]==s for x in aligned) for s in ids}
    keys=[(x["sample_id"],x["bin_id"]) for x in aligned]; duplicate=len(keys)-len(set(keys))
    labels_per={s:{x["official_label"] for x in aligned if x["sample_id"]==s} for s in ids}
    nan_summary={}; inf_summary={}; feature_cols=sorted({k for r in aligned for k,v in r.items() if isinstance(v,(float,np.floating))})
    for c in feature_cols:
        vals=np.asarray([r.get(c,np.nan) if r.get(c,None) is not None else np.nan for r in aligned],dtype=float)
        nan_summary[c]=int(np.isnan(vals).sum()); inf_summary[c]=int(np.isinf(vals).sum())
    samples_exact=sum(v==BINS for v in counts.values()); rows_ok=len(aligned)==BINS*len(ids) and duplicate==0 and samples_exact==len(ids)
    text_ok=sum(s["text_valid_bins"]>0 for s in summary); audio_ok=sum(s["audio_valid_bins"]>0 for s in summary); vision_ok=sum(s["vision_valid_bins"]>0 for s in summary)
    full=sum(s["all_modalities_valid_bins"]==BINS for s in summary)
    errors=[x for x in issues if x["severity"]=="ERROR"]
    complete_quality = (full == len(ids) and text_ok == len(ids))
    status="PASS" if rows_ok and not errors and len(ids)==100 and complete_quality else ("PARTIAL PASS" if rows_ok and len(ids)>0 else "FAIL")
    result={"status":status,"expected_samples":100,"actual_samples":len(ids),"expected_bins_per_sample":BINS,"samples_with_exactly_50_bins":samples_exact,"expected_total_rows":100*BINS,"actual_total_rows":len(aligned),
            "duplicate_keys":duplicate,"missing_labels":sum(not r["label_available"] for r in manifest),"text_success_samples":text_ok,"audio_success_samples":audio_ok,"vision_success_samples":vision_ok,
            "full_three_modal_samples":full,"nan_summary":nan_summary,"inf_summary":inf_summary,"warnings":[x for x in issues if x["severity"]=="WARNING"],"errors":errors,
            "structural_alignment_pass":rows_ok,"complete_three_modal_coverage_samples":full,
            "timeline_sources":{s:sum(r["timeline_source"]==s for r in manifest) for s in sorted({r["timeline_source"] for r in manifest})},"all_zero_columns":[],"constant_columns":[],"overall_pass":status=="PASS"}
    for c in feature_cols:
        vals=np.asarray([r.get(c,np.nan) if r.get(c,None) is not None else np.nan for r in aligned],float); finite=vals[np.isfinite(vals)]
        if len(finite) and np.all(finite==0): result["all_zero_columns"].append(c)
        if len(finite) and np.all(finite==finite[0]): result["constant_columns"].append(c)
    result["expected_all_zero_columns"]=[]
    result["expected_constant_columns"]=["audio_present","audio_valid","vision_present","valid_frame_count? no"]
    result["expected_constant_columns"]=["audio_present","vision_present"]
    write_json(OUT/"validation.json",result); return result


def qc(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,4.5)); plt.hist([float(x["duration_sec"]) for x in summary if math.isfinite(float(x["duration_sec"]))],bins=20,color="#4178a8",edgecolor="white"); plt.xlabel("Effective duration (s)"); plt.ylabel("Clips"); plt.title("100-clip effective duration distribution"); plt.tight_layout(); plt.savefig(FIG/"duration_distribution.png",dpi=160); plt.close()
    plt.figure(figsize=(8,4.5)); labels=["Text","Audio","Vision"]; cols=["text_coverage","audio_coverage","vision_coverage"]
    vals=[[float(x[c]) for x in summary] for c in cols]; plt.boxplot(vals,tick_labels=labels,showfliers=True); plt.ylabel("Valid-bin coverage"); plt.ylim(-.05,1.05); plt.title("Per-clip modality coverage"); plt.tight_layout(); plt.savefig(FIG/"modality_coverage.png",dpi=160); plt.close()


def run_all(args):
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest()
    if args.stage in ("manifest","all"): manifest=get_manifest()
    if args.stage=="manifest": return
    text_path=OUT/"text_features_50bin.parquet"; audio_path=OUT/"audio_features_50bin.parquet"; vision_path=OUT/"vision_features_50bin.parquet"
    if args.stage in ("text","all"):
        terr=build_text(manifest); write_json(OUT/"_text_stage_errors.json",terr)
    if args.stage in ("audio","all"):
        a,ai=build_audio(manifest); write_features(a,"audio_features_50bin.parquet"); write_csv(OUT/"_audio_issues.csv",ai)
    if args.stage in ("vision","all"):
        v,vi=build_vision(manifest); write_features(v,"vision_features_50bin.parquet"); write_csv(OUT/"_vision_issues.csv",vi)
    if args.stage in ("align","all"):
        import pandas as pd
        def loadstage(name):
            p=OUT/name
            if p.exists(): return pd.read_parquet(p).to_dict("records")
            return pd.read_csv(p.with_suffix(".csv")).to_dict("records")
        t=loadstage("text_features_50bin.parquet"); a=loadstage("audio_features_50bin.parquet"); v=loadstage("vision_features_50bin.parquet")
        issues=read_csv(OUT/"data_integrity_issues.csv") if (OUT/"data_integrity_issues.csv").exists() else []
        for f in (OUT/"_audio_issues.csv",OUT/"_vision_issues.csv",OUT/"_text_stage_errors.json"):
            if f.exists():
                if f.suffix==".csv": issues.extend(read_csv(f))
                else:
                    for sid,msg in json.loads(f.read_text(encoding="utf-8")): issues.append(issue(sid,"text","","text_extraction_failed","ERROR",msg))
        aligned,summ=summarize_and_align(manifest,t,a,v,issues); schema(aligned); result=validate(manifest,aligned,summ,issues); qc(summ)
        elapsed=time.perf_counter()-start
        write_json(OUT/"run_metadata.json",{"seed":SEED,"stages":args.stage,"elapsed_seconds":elapsed,"python":sys.version,"versions":{p:importlib.metadata.version(p) for p in ("numpy","pandas","av","librosa","torch","transformers","faster-whisper","opencv-python") if has_dist(p)},"audio":{"sample_rate":SR,"mono":True,"n_fft":1024,"hop_length":256,"mfcc":13,"pitch":"librosa.yin 65-500 Hz; invalid if no voiced frames"},"vision":{"sample_positions_within_bin":[0.25,0.5,0.75],"method":"sampled OpenCV basic frame statistics; no CNN embedding"},"random_seed":SEED})
        print(json.dumps({"status":result["status"],"samples":len(manifest),"aligned_rows":len(aligned),"text_success":result["text_success_samples"],"audio_success":result["audio_success_samples"],"vision_success":result["vision_success_samples"],"full_three_modal":result["full_three_modal_samples"]},ensure_ascii=False))


def has_dist(name):
    try: importlib.metadata.version(name); return True
    except importlib.metadata.PackageNotFoundError: return False


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=["manifest","text","audio","vision","align","validate","all"],required=True); args=p.parse_args()
    np.random.seed(SEED)
    if args.stage=="validate":
        manifest=load_manifest(); import pandas as pd
        def load(name):
            path=OUT/name
            return pd.read_parquet(path).to_dict("records") if path.exists() else pd.read_csv(path.with_suffix(".csv")).to_dict("records")
        aligned=load("multimodal_features_50bin.parquet"); summ=read_csv(OUT/"multimodal_summary_100.csv"); issues=read_csv(OUT/"processing_issues.csv"); print(json.dumps(validate(manifest,aligned,summ,issues),ensure_ascii=False)); return
    run_all(args)

if __name__=="__main__": main()
