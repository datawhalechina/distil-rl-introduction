# Chapter 0. Prelude 

## What This Is  

As someone who's been learning reinforcement learning (RL) myself, I've been on the hunt for a while to find material that hits just the right spot. There are plenty of great resources out there, but most of them either dive straight into advanced algorithms or fail to connect the dots between different concepts in a way that makes sense for beginners.  

*Reinforcement Learning: An Introduction* by Richard Sutton (I'll just refer to it as the RL Intro book from now on) is regarded as the bible of reinforcement learning. It's an incredible resource, no doubt about it, but reading it cover-to-cover takes a ton of effort. And honestly, most learners - whether they're researchers or practitioners - don't need every single detail in the book to grasp the core ideas, understand classical algorithms, and start applying or even developing new ones.  

That's exactly why I created this tutorial. After spending some time organizing my learning notes while working through the RL Intro book, I started thinking that maybe there is a faster, more efficient way to absorb the key knowledge from this book without having to read every word. That is how this project was born - a streamlined "knowledge vault" of reinforcement learning based on the RL Intro book.  

Out of all the chapters in the book, I have **handpicked the most essential ones to help you build a solid understanding**. I used the structure of [this Coursera specialization](https://www.coursera.org/specializations/reinforcement-learning), taught by members of Sutton’s research group, as a guide. This knowledge vault blends the best parts of the RL Intro book and the online course, switching between the two formats to give you the easiest path to learning based on my own experience.  

**Starting with my personal notes**, I have added introductory explanations to each chapter and smoothed out the flow of the content to make it more approachable. But let's be real- I am just one person who does not consider myself the "author" of this project but rather a "knowledge curator". So like any creator would, I'd welcome any feedback you have to help improve the quality of this tutorial!

## How to Read  

- **Structure of This Tutorial**

    The chapters in this tutorial are carefully selected and reorganized based on the structure of the paired Coursera course. One notable difference is that Chapter 7, "n-step Bootstrapping," from the RL Intro book has been excluded.  

    The selected chapters are grouped into two main categories, as outlined in the RL Intro book: **Tabular Solution Methods** and **Approximation Solution Methods**. You can find these categories listed on the homepage or in the sidebar for easy navigation.  

    The next chapter gives a quick introduction to reinforcement learning (RL). Chapters 2 through 7 focus on RL in what's called the *tabular setting*. Starting from Chapter 8, we shift gears to focus exclusively on function approximation methods. The final chapter isn't part of the RL Intro book - it’s an additional section I’ve added to provide insights into more modern developments in RL.  

- **Style** 

    - **Structure of Each Chapter**: Most of the content in each chapter is presented in a bullet-point format. To help you quickly grasp the purpose of each section, key chunks of text are introduced with a bolded word or phrase (like the one you’re reading now). Think of it as a mini headline for the paragraph.  

    - **Optional Content**: Sections or paragraphs marked with a $\star$ are optional. Feel free to skip them if you’re short on time or just want to focus on the essentials.  

    - **Concepts to Be Defined**: Any terms or concepts that need further explanation are written in $\textit{italicized text}$.  

    - **Notations**:  

        - Capital letters with time steps as subscripts represent random variables, such as $A_t$.  

            - In algorithm presentations, capital letters (without subscripts) are used to denote realizations of random variables. 

        - Lowercase letters (with or without subscripts) represent specific instances of the respective random variables. For example, $s$ is a specific state from the set $S$, and $s_t$ refers to the state at time step $t$ (similarly for $s_0$, $s_1$, etc.).  

- **Learning Advice**  

    - **How I Blend the Book and Online Course (Videos)**: My goal was to combine the best parts of the RL Intro book and the Coursera lecture videos to make learning easier. Here’s how I did it:  

        - For complex math derivations, I provide both text explanations and video walkthroughs. Use whichever format works better for you.  

        - For step-by-step demonstrations, I’ve included screenshots from the videos to speed up reading. The full lecture videos are usually provided as optional materials if you want to dive deeper.  

        - For problem examples (from introduction to solution), I mostly rely on videos because they tend to explain things more clearly. These are embedded as clickable screenshots, and I sometimes add quick notes underneath to highlight key points.  

    - **How You Can Augment Your Learning**:  Consider using a language model (LLM) product that supports web searches to act as your personal assistant. Have it guide you through tricky concepts. Simple as that!  


## Potential Alternatives

To help fellow reinforcement learning (RL) enthusiasts learn more effectively, I've put together this tutorial. But everyone learns in their own way, so here are a few other great resources (in my opinion) that might help you on your RL journey:

- [A similar online RL book written by Professor Tim Miller](https://gibberblot.github.io/rl-notes/index.html)

- [OpenAI spinning up](https://spinningup.openai.com/en/latest/index.html) (More suitable for learners who have experience with Deel Learning to some level)

- [Huggingface Deep RL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) (You can actually earn a certificate here)

- [David Silver's RL course at UCL](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) (My personal favorite)

- Some cheet-sheet-like RL study notes:

    - [PDF notes from Scott Jeen](https://enjeeneer.io/sutton_and_barto/rl_notes.pdf)

    - [Notes from Blog IndoML](https://indoml.com/2018/02/14/study-notes-reinforcement-learning-an-introduction/#:~:text=The%20main%20elements%20of%20RL,a%20model%20of%20the%20environment.&text=The%20learner%2Fdecision%20maker%20being%20trained.)

    - [Notes from Luciano Strika](https://strikingloo.github.io/wiki/reinforcement-learning-sutton)

- [Github awesome-deep-rl](https://github.com/kengz/awesome-deep-rl)

Lastly, the learning curve in RL can feel incredibly steep at the beginning due to its inherently complex nature. Ask further on reddit, search for other courses on some learning **platform** like Coursera, you'll always find what is best for you. 

## Acknowledgements and Urges

My deep appreciation to [Marvin Schweizer](https://github.com/mschweizer) for providing valuable feedback on this project!

The project could definitely use more polish and I'd love to hear from you. Feel free to open issues on GitHub to share your thoughts- your input will go a long way in making this resource better for everyone.