"""
智能提示词构建器
根据用户选择的军事目标、天气、场景自动构建高质量提示词
"""

from typing import Dict, List, Tuple
import random
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """提示词模板"""
    base: str
    quality_enhancers: List[str]
    style_modifiers: List[str]
    technical_terms: List[str]

class PromptBuilder:
    """智能提示词构建器"""
    
    def __init__(self):
        """初始化提示词构建器"""
        self._init_templates()
        self._init_negative_prompts()
    
    def _init_templates(self):
        """初始化提示词模板"""
        
        # 军事目标模板（与传统生成页面保持一致）
        self.military_targets = {
            "坦克": PromptTemplate(
                base="military tank, armored vehicle, main battle tank",
                quality_enhancers=[
                    "highly detailed", "photorealistic", "8k resolution",
                    "professional military photography", "sharp focus"
                ],
                style_modifiers=[
                    "camouflage pattern", "weathered metal", "battle-worn",
                    "dust and dirt", "realistic textures", "military green"
                ],
                technical_terms=[
                    "turret", "cannon", "tracks", "armor plating",
                    "periscope", "machine gun", "reactive armor"
                ]
            ),
            "战机": PromptTemplate(
                base="military fighter jet, combat aircraft, warplane",
                quality_enhancers=[
                    "highly detailed", "photorealistic", "8k resolution",
                    "aviation photography", "sharp focus", "dynamic angle"
                ],
                style_modifiers=[
                    "sleek design", "metallic surface", "jet engines",
                    "cockpit canopy", "military markings", "aerodynamic"
                ],
                technical_terms=[
                    "wings", "fuselage", "afterburner", "missiles",
                    "radar dome", "landing gear", "air intake"
                ]
            ),
            "舰艇": PromptTemplate(
                base="military warship, naval vessel, battleship",
                quality_enhancers=[
                    "highly detailed", "photorealistic", "8k resolution",
                    "naval photography", "sharp focus", "maritime scene"
                ],
                style_modifiers=[
                    "steel hull", "naval gray paint", "deck equipment",
                    "radar arrays", "communication towers", "weathered metal"
                ],
                technical_terms=[
                    "bridge", "gun turrets", "missile launchers",
                    "radar systems", "deck", "superstructure", "hull"
                ]
            )
        }
        
        # 天气条件模板（与传统生成页面保持一致）
        self.weather_conditions = {
            "雨天": {
                "base": "rainy weather, heavy rain, storm",
                "atmosphere": ["dark clouds", "water droplets", "wet surfaces", "puddles"],
                "lighting": ["overcast sky", "diffused lighting", "gray atmosphere"],
                "effects": ["rain drops", "water reflections", "misty air", "stormy sky"]
            },
            "雪天": {
                "base": "snowy weather, winter scene, snowfall",
                "atmosphere": ["falling snow", "snow-covered ground", "white landscape"],
                "lighting": ["soft winter light", "bright snow reflection", "cold atmosphere"],
                "effects": ["snowflakes", "frost", "icy surfaces", "winter conditions"]
            },
            "大雾": {
                "base": "foggy weather, heavy fog, misty conditions",
                "atmosphere": ["thick fog", "low visibility", "misty air", "hazy atmosphere"],
                "lighting": ["diffused light", "muted colors", "soft shadows"],
                "effects": ["fog banks", "reduced visibility", "atmospheric haze", "mysterious mood"]
            },
            "夜间": {
                "base": "night scene, nighttime, dark environment",
                "atmosphere": ["dark sky", "night atmosphere", "low light conditions"],
                "lighting": ["artificial lighting", "spotlights", "night vision", "dramatic shadows"],
                "effects": ["night operations", "tactical lighting", "moonlight", "starry sky"]
            }
        }
        
        # 地形场景模板（与传统生成页面保持一致）
        self.scene_environments = {
            "城市": {
                "base": "urban environment, city setting, metropolitan area",
                "background": ["city buildings", "urban landscape", "concrete structures"],
                "details": ["streets", "infrastructure", "urban warfare", "civilian buildings"],
                "atmosphere": ["urban setting", "metropolitan", "city environment"]
            },
            "岛屿": {
                "base": "island setting, coastal environment, maritime location",
                "background": ["ocean view", "coastal landscape", "island terrain"],
                "details": ["beach", "rocky shores", "tropical vegetation", "sea breeze"],
                "atmosphere": ["island paradise", "coastal defense", "maritime operations"]
            },
            "乡村": {
                "base": "rural environment, countryside setting, natural landscape",
                "background": ["open fields", "natural terrain", "rural landscape"],
                "details": ["grass fields", "trees", "natural environment", "country roads"],
                "atmosphere": ["peaceful countryside", "rural operations", "natural setting"]
            }
        }
        
        # 通用质量增强词
        self.quality_enhancers = [
            "masterpiece", "best quality", "ultra detailed", "8k wallpaper",
            "extremely detailed", "high resolution", "sharp focus",
            "professional photography", "realistic", "photorealistic",
            "cinematic lighting", "perfect composition", "award winning photo"
        ]
        
        # 技术参数增强词
        self.technical_enhancers = [
            "shot with Canon EOS R5", "85mm lens", "f/2.8 aperture",
            "professional camera", "DSLR photography", "studio lighting",
            "perfect exposure", "color grading", "post-processing"
        ]
    
    def _init_negative_prompts(self):
        """初始化负面提示词"""
        self.negative_prompts = {
            "quality": [
                "low quality", "worst quality", "low resolution", "blurry",
                "out of focus", "pixelated", "jpeg artifacts", "compression artifacts",
                "noise", "grainy", "distorted", "deformed"
            ],
            "anatomy": [
                "bad anatomy", "poorly drawn", "malformed", "disfigured",
                "mutated", "extra limbs", "missing parts", "broken"
            ],
            "style": [
                "cartoon", "anime", "manga", "drawing", "painting",
                "sketch", "illustration", "3d render", "cgi"
            ],
            "content": [
                "text", "watermark", "signature", "logo", "copyright",
                "username", "artist name", "title", "border"
            ],
            "inappropriate": [
                "nsfw", "nude", "sexual", "violence", "gore",
                "disturbing", "offensive", "inappropriate"
            ]
        }
    
    def build_prompt(
        self,
        military_target: str,
        weather: str,
        scene: str,
        style_strength: float = 0.7,
        technical_detail: float = 0.8,
        quality_boost: bool = True
    ) -> Tuple[str, str]:
        """
        构建完整的提示词
        
        Args:
            military_target: 军事目标类型
            weather: 天气条件
            scene: 场景环境
            style_strength: 风格强度 (0.0-1.0)
            technical_detail: 技术细节程度 (0.0-1.0)
            quality_boost: 是否添加质量增强词
            
        Returns:
            Tuple[str, str]: (正面提示词, 负面提示词)
        """
        
        # 构建正面提示词
        positive_parts = []
        
        # 1. 主体描述
        if military_target in self.military_targets:
            target_template = self.military_targets[military_target]
            positive_parts.append(target_template.base)
            
            # 添加技术细节
            if technical_detail > 0.5:
                tech_terms = random.sample(
                    target_template.technical_terms,
                    min(3, len(target_template.technical_terms))
                )
                positive_parts.extend(tech_terms)
            
            # 添加风格修饰
            if style_strength > 0.3:
                style_mods = random.sample(
                    target_template.style_modifiers,
                    min(2, len(target_template.style_modifiers))
                )
                positive_parts.extend(style_mods)
        
        # 2. 天气条件
        if weather in self.weather_conditions:
            weather_template = self.weather_conditions[weather]
            positive_parts.append(weather_template["base"])
            
            # 添加大气效果
            atmosphere = random.sample(weather_template["atmosphere"], 2)
            positive_parts.extend(atmosphere)
            
            # 添加光照效果
            lighting = random.choice(weather_template["lighting"])
            positive_parts.append(lighting)
            
            # 添加特效
            if style_strength > 0.5:
                effects = random.sample(weather_template["effects"], 2)
                positive_parts.extend(effects)
        
        # 3. 场景环境
        if scene in self.scene_environments:
            scene_template = self.scene_environments[scene]
            positive_parts.append(scene_template["base"])
            
            # 添加背景描述
            background = random.choice(scene_template["background"])
            positive_parts.append(background)
            
            # 添加环境细节
            if technical_detail > 0.4:
                details = random.sample(scene_template["details"], 2)
                positive_parts.extend(details)
        
        # 4. 质量增强
        if quality_boost:
            quality_terms = random.sample(self.quality_enhancers, 3)
            positive_parts.extend(quality_terms)
            
            # 添加技术参数
            if technical_detail > 0.7:
                tech_terms = random.sample(self.technical_enhancers, 2)
                positive_parts.extend(tech_terms)
        
        # 构建负面提示词
        negative_parts = []
        for category, terms in self.negative_prompts.items():
            selected_terms = random.sample(terms, min(3, len(terms)))
            negative_parts.extend(selected_terms)
        
        # 组合提示词
        positive_prompt = ", ".join(positive_parts)
        negative_prompt = ", ".join(negative_parts)
        
        return positive_prompt, negative_prompt
    
    def build_custom_prompt(
        self,
        base_prompt: str,
        military_target: str = None,
        weather: str = None,
        scene: str = None,
        enhance_quality: bool = True
    ) -> Tuple[str, str]:
        """
        基于自定义基础提示词构建
        
        Args:
            base_prompt: 基础提示词
            military_target: 军事目标（可选）
            weather: 天气条件（可选）
            scene: 场景环境（可选）
            enhance_quality: 是否增强质量
            
        Returns:
            Tuple[str, str]: (正面提示词, 负面提示词)
        """
        parts = [base_prompt]
        
        # 添加可选的增强元素
        if military_target and military_target in self.military_targets:
            target_template = self.military_targets[military_target]
            parts.extend(random.sample(target_template.style_modifiers, 2))
        
        if weather and weather in self.weather_conditions:
            weather_template = self.weather_conditions[weather]
            parts.append(weather_template["base"])
            parts.extend(random.sample(weather_template["atmosphere"], 1))
        
        if scene and scene in self.scene_environments:
            scene_template = self.scene_environments[scene]
            parts.append(scene_template["base"])
        
        if enhance_quality:
            quality_terms = random.sample(self.quality_enhancers, 2)
            parts.extend(quality_terms)
        
        # 构建负面提示词
        negative_parts = []
        for category in ["quality", "style", "content"]:
            if category in self.negative_prompts:
                terms = random.sample(self.negative_prompts[category], 2)
                negative_parts.extend(terms)
        
        positive_prompt = ", ".join(parts)
        negative_prompt = ", ".join(negative_parts)
        
        return positive_prompt, negative_prompt
    
    def get_prompt_suggestions(self, military_target: str) -> Dict[str, List[str]]:
        """
        获取提示词建议
        
        Args:
            military_target: 军事目标类型
            
        Returns:
            Dict[str, List[str]]: 提示词建议分类
        """
        if military_target not in self.military_targets:
            return {}
        
        template = self.military_targets[military_target]
        
        return {
            "基础描述": [template.base],
            "质量增强": template.quality_enhancers,
            "风格修饰": template.style_modifiers,
            "技术术语": template.technical_terms,
            "通用质量": self.quality_enhancers[:5],
            "技术参数": self.technical_enhancers[:3]
        }
    
    def optimize_prompt(self, prompt: str) -> str:
        """
        优化提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 优化后的提示词
        """
        # 移除重复词汇
        words = [word.strip() for word in prompt.split(",")]
        unique_words = []
        seen = set()
        
        for word in words:
            if word.lower() not in seen and word:
                unique_words.append(word)
                seen.add(word.lower())
        
        # 重新排序：重要词汇前置
        important_keywords = ["military", "tank", "fighter", "warship", "detailed", "photorealistic"]
        
        important_words = []
        other_words = []
        
        for word in unique_words:
            if any(keyword in word.lower() for keyword in important_keywords):
                important_words.append(word)
            else:
                other_words.append(word)
        
        # 组合优化后的提示词
        optimized_prompt = ", ".join(important_words + other_words)
        
        return optimized_prompt
