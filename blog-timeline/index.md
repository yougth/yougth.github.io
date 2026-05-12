---
layout: default
---

{% assign sorted_posts = site.posts | sort: 'date' | reverse %}
{% assign post_count = sorted_posts | size %}

<div class="timeline-wrap">

  {% for post in sorted_posts %}
    <div class="timeline-item">
      <div class="timeline-date">{{ post.date | date: "%Y" }} · {{ post.date | date: "%m-%d" }}</div>
      <div class="timeline-title">
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </div>
      <div class="timeline-tags">
        {% for tag in post.tags %}
          {% assign tag_color = site.tag_colors[tag] | default: "#78716c" %}
          <span class="card-tag" style="background: {{ tag_color }}">{{ tag }}</span>
        {% endfor %}
      </div>
      {% if post.subtitle %}
        <div class="timeline-subtitle">{{ post.subtitle }}</div>
      {% endif %}
    </div>
  {% endfor %}

</div>
