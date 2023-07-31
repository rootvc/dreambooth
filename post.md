# The Nitty-Gritty of Building an AI-Powered Photo Booth at RootVC

Greetings from RootVC, where venture capital meets the hacker spirit! As a team of engineers at heart, we love diving into inventive projects that challenge the status quo. The catalyst for our next project was a serendipitous find by our team member, Kane—a retro photo booth that was begging for a 21st-century upgrade.

In this blog post, we're pulling back the curtain on our journey of transforming this classic relic into an AI-powered dynamo we call the IRL Dreambooth project. It's a tale of challenges, clever solutions, and a generous dose of creative engineering.

## The IRL Dreambooth Project – Where Nostalgia Meets Innovation

The IRL Dreambooth project was about giving a tech makeover to the nostalgic photo booth. The goal was to inject the static nature of traditional photo booths with a dose of generative AI, providing users with a personalized, dynamic experience.

The journey was an exciting one, filled with hurdles each step of the way. From tracking down the perfect retro photo booth to fine-tuning an AI model to align with our vision, every step was a challenge waiting to be tackled.

## Overcoming Retro Tech and Syncing Snags

Our first hurdle was finding a retro-style photo booth that could be the canvas for our creativity. After some extensive online treasure hunt and a dash of luck, Kane unearthed the perfect fit for our project.

![Image Placeholder: The Retro Photo Booth](#)

The booth came with an old Windows 95 system that was completely untenable. Our solution was a good ol' rip-and-replace: we swapped the box for a Mac Mini, and managed to find the same jank photobooth software to run on it.

![Image Placeholder: The Upgraded System](#)

This was, of course, after we managed to wipe out an entire night's worth of photos on the old box. We used an S3Sync software to upload the photos to an S3 bucket after they were taken. However, this was a two-way street. When we tried to clear the S3 cache, it wiped our local photos clean, erasing all snapshots taken during a holiday party. Let's just say, it was a night that taught us an unforgettable lesson on syncing systems!

## Taming the AI Beast

Fine-tuning the Dreambooth model was akin to solving a Rubik's cube blindfolded. We started with tutorials on [Stable Diffusion](https://arxiv.org/abs/2112.10752) (SD) 1.5, which had our training times clocking in at around an hour. But, time was a luxury we didn't have. After some optimization efforts by Lee, our resident AI whisperer, we managed to shave off the training time to 20 minutes. The real game-changer was Yasyf's work, introducing textual inversion and [Low-Rank Adaptation](https://arxiv.org/abs/2106.04482) of Stable Diffusion (LORA) technique, bringing down the training time to a mere minute and a half.

Inference was performed via ControlNet, starting with a depth mask generated from the original images. After multiple rounds of testing and tweaking, we finally had a model that could complete training and inference within 2.5 minutes. For those interested in a deeper dive into these techniques, our [GitHub repo](#) offers a detailed look at our experiments and findings.

![Image Placeholder: Sample Image Output (Inference)](#)

Crafting creative prompts that could generate a wide array of engaging images was a task that required a blend of technical acumen and creativity. With quite a bit of testing, we found that people were not fans of the model making them look old. This insight led us to discourage the model from doing so, along with other unflattering terms such as ugly and distorted.

## Delivering AI Magic with Inngest

To ensure smooth delivery of our AI magic, we relied on [Inngest](https://www.inngest.com/), an orchestration tool. From handling rate limiting to retrying failed jobs, Inngest functioned as the conductor, orchestrating a seamless workflow for the Dreambooth experience.

## The Journey Continues...

The IRL Dreambooth project was more than just a project. It was a testament to the potential of Generative AI and the transformative power of creative engineering. We've managed to couple a retro photo booth with AI, delivering a unique and engaging user experience.

Looking ahead, we're excited about the possibilities of incorporating more user input to customize prompts further. Imagine a photo booth picture with your favorite animal perched on your shoulder!

![Image Placeholder: Final Photo Booth Picture with User's Favourite Animal](#)

We hope this post has offered a peek into our engineering journey and the lessons we've learned along the way. As we continue our exploration into the potential of generative AI and its myriad applications, we invite you, fellow engineers and founders, to join our conversation. Got an interesting idea or a thought-provoking question? We'd love to hear it! Stay tuned for more fun projects and insightful discussions from the RootVC team.
