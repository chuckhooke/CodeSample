import torch
import torch.nn.functional as F

# Prompts for pos/neg tunnel walls
POSITIVE_PROMPTS = [
    "intact tunnel wall",
    "undamaged tunnel surface",
    "stable tunnel wall",
    "normal tunnel wall",
    "well-maintained tunnel structure with no visible damage",
    "smooth and crack-free tunnel wall",
    "tunnel wall without signs of damage or instability",
    "an undisturbed, structurally sound tunnel wall"
]

NEGATIVE_PROMPTS = [
    "cracked tunnel wall",
    "damaged tunnel surface",
    "destroyed tunnel wall",
    "unstable tunnel wall",
    "tunnel wall with visible cracks and signs of erosion",
    "deteriorating tunnel wall with structural weakness",
    "surface damage on tunnel wall due to heavy impact",
    "severely damaged tunnel wall with risk of collapse"
]

TUNNEL_PROMPTS = POSITIVE_PROMPTS + NEGATIVE_PROMPTS


# Encode image
def encode_image(clip_model, pre_image, post_image):
    with torch.no_grad():
        pre_fea = clip_model.encode_image(pre_image).float()
        post_fea = clip_model.encode_image(post_image).float()
        # Normalize them
        pre_fea = F.normalize(pre_fea, p=2, dim=-1)
        post_fea = F.normalize(post_fea, p=2, dim=-1)
    return pre_fea, post_fea


# Encode prompt
def encode_prompt(clip_model, prompts):
    with torch.no_grad():
        tokens = clip_model.tokenize(prompts).cuda()
        text_fea = clip_model.encode_text(tokens).float()
        # Normalize
        text_fea = F.normalize(text_fea, p=2, dim=-1)
    return text_fea


# Similarity between image and prompt
def similarity(image_fea, text_fea):
    similarity = image_fea @ text_fea.T
    return similarity


# Similarity for pos and neg prompts
def assess_condition(sim_pre, sim_post, num_pos):
    avg_sim_pre_pos = sim_pre[0, :num_pos].mean().item()
    avg_sim_pre_neg = sim_pre[0, num_pos:].mean().item()
    avg_sim_post_pos = sim_post[0, :num_pos].mean().item()
    avg_sim_post_neg = sim_post[0, num_pos:].mean().item()

    # Sim change
    delta_pos = avg_sim_post_pos - avg_sim_pre_pos
    delta_neg = avg_sim_post_neg - avg_sim_pre_neg

    if delta_neg > delta_pos:
        label = 1  # Damaged
    else:
        label = 0  # Undamaged
    return label


def evaluate_tunnel_condition(clip_model, pre_image, post_image):
    # Encode image
    pre_fea, post_fea = encode_image(clip_model, pre_image, post_image)
    # Encode prompt
    text_fea = encode_prompt(clip_model, TUNNEL_PROMPTS)
    # Compute sim
    sim_pre = similarity(pre_fea, text_fea)
    sim_post = similarity(post_fea, text_fea)
    # Assess the condition
    label = assess_condition(sim_pre, sim_post, len(POSITIVE_PROMPTS_TUNNEL))
    return label
