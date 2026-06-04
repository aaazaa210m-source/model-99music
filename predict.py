from cog import BasePredictor, Input, Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch

class Predictor(BasePredictor):
    def setup(self):
        """تحميل الأوزان والنموذج المدرب الجاهز إلى الذاكرة عند إقلاع السيرفر"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)

    def predict(
        self,
        prompt: str = Input(description="وصف اللحن أو الأغنية المطلوبة لتوليدها"),
        duration: int = Input(description="مدة المقطع الصوتي بالثواني", default=10)
    ) -> Path:
        """استقبال الأوامر وتوليد ملف الصوت بنجاح"""
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # حساب عدد التوكنز المطلوبة بناءً على الثواني لضبط مدة الأغنية
        max_tokens = int(duration * 51.2)
        
        # توليد الأغنية من النموذج المعرف في بيئة السيرفر
        audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens)
        audio_data = audio_values[0, 0].cpu().numpy()
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        
        # حفظ الناتج في مسار مؤقت ليقوم سيرفر Replicate برباطه وإعطائك الرابط النهائي
        output_path = "/tmp/output.wav"
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data)
        
        return Path(output_path)
