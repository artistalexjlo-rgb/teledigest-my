/* Октагон: рисуем path под РЕАЛЬНЫЙ размер коробки (не растягиваем готовый контур).
   Скос фикс-px + скругление строятся в настоящих px → ровный 45° на любом размере.
   ResizeObserver ловит изменение размера от чего угодно (контент, перенос, шрифт). */
(function(){
  function pts(w,h,c){return[[c,0],[w-c,0],[w,c],[w,h-c],[w-c,h],[c,h],[0,h-c],[0,c]];}
  function rp(p,r){var n=p.length,d="";function u(a,b){var dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy)||1;return[dx/L,dy/L];}
    for(var i=0;i<n;i++){var p1=p[i],p0=p[(i-1+n)%n],p2=p[(i+1)%n],u0=u(p1,p0),u2=u(p1,p2),
      l0=Math.hypot(p0[0]-p1[0],p0[1]-p1[1]),l2=Math.hypot(p2[0]-p1[0],p2[1]-p1[1]),rr=Math.min(r,l0/2,l2/2),
      a=[p1[0]+u0[0]*rr,p1[1]+u0[1]*rr],b=[p1[0]+u2[0]*rr,p1[1]+u2[1]*rr];
      d+=(i?"L":"M")+a[0].toFixed(2)+","+a[1].toFixed(2)+"Q"+p1[0].toFixed(2)+","+p1[1].toFixed(2)+" "+b[0].toFixed(2)+","+b[1].toFixed(2);}
    return d+"Z";}
  function draw(svg){var w=Math.round(svg.clientWidth),h=Math.round(svg.clientHeight);if(w<4||h<4)return;
    var c=Math.min(parseFloat(svg.dataset.cant)||8,w/2-1,h/2-1),r=parseFloat(svg.dataset.round);if(isNaN(r))r=3;
    svg.setAttribute("viewBox","0 0 "+w+" "+h);svg.firstElementChild.setAttribute("d",rp(pts(w,h,c),r));}
  var ro=window.ResizeObserver?new ResizeObserver(function(es){for(var i=0;i<es.length;i++)draw(es[i].target);}):null;
  function init(){var l=document.querySelectorAll("svg.oct");for(var i=0;i<l.length;i++){draw(l[i]);if(ro)ro.observe(l[i]);}}
  if(document.readyState!=="loading")init();else document.addEventListener("DOMContentLoaded",init);
  if(!ro){window.addEventListener("load",init);var t;window.addEventListener("resize",function(){clearTimeout(t);t=setTimeout(init,120);});}
})();

/* ПОИСК ПО ЗАГОЛОВКАМ (шаг 7). Индекс языка тянем ОДИН раз по первому вводу:
   ru — 3124 заголовка, 352 КБ (77 КБ в gzip). Инлайн в страницу означал бы те же
   352 КБ × 41 630 страниц. Ни ключей, ни сервера, ни продукта — только статика.
   Enter уводит на /<язык>/find/?s=…: это обычная навигация, поэтому запрос виден в
   логе веб-сервера, и мы наконец знаем, ЧТО люди спрашивают. */
(function(){
  var q=document.getElementById("gq"); if(!q) return;
  var sg=document.getElementById("gsugg"), IDX=null, loading=false;
  function go(){var v=q.value.trim(); if(v) location.href=q.dataset.find+"?s="+encodeURIComponent(v);}
  function show(){
    var v=q.value.trim().toLowerCase();
    if(!v||!IDX){sg.style.display="none";return;}
    /* Страны — вперёд: на главной человек чаще ищет страну, а индекс отсортирован по
       алфавиту, и «гре» иначе выдало бы шесть заголовков раньше самой Греции.
       Хаб страны узнаём по адресу: /<язык>/<страна>/ — ровно два сегмента. */
    var hits=[],i,r,seg;
    for(i=0;i<IDX.length;i++){
      r=IDX[i];
      if(r[0].toLowerCase().indexOf(v)<0) continue;
      seg=r[1].split("/").filter(function(x){return x;}).length;
      hits.push([seg===2?0:1,r]);
      if(hits.length>=40) break;
    }
    hits.sort(function(a,b){return a[0]-b[0];});
    var m=hits.slice(0,8).map(function(h){return h[1];});
    sg.innerHTML = m.length
      ? m.map(function(r){return '<a href="'+r[1]+'">'+r[0]+"</a>";}).join("")
      : '<a class="nores">'+q.dataset.none+"</a>";
    sg.style.display="block";
  }
  function load(){
    if(IDX||loading) return; loading=true;
    fetch(q.dataset.index).then(function(r){return r.json();}).then(function(j){IDX=j;show();})
      .catch(function(){loading=false;});  /* индекс не доехал — поле просто уводит на find */
  }
  q.addEventListener("focus",load);
  q.addEventListener("input",function(){load();show();});
  q.addEventListener("keydown",function(e){if(e.key==="Enter"){e.preventDefault();go();}});
  document.addEventListener("click",function(e){
    if(!e.target.closest(".gsearch")&&e.target.id!=="gq") sg.style.display="none";
  });
})();

/* Страница поиска /<язык>/find/: рисуем результаты из того же индекса. На статике
   иначе никак, а генерацию на сайте канон запрещает. */
(function(){
  var box=document.getElementById("results"); if(!box) return;
  var s=new URLSearchParams(location.search).get("s")||"";
  var inp=document.getElementById("fq"); if(inp) inp.value=s;  /* запрос виден в самом поле */
  if(!s){box.innerHTML="";return;}
  fetch(box.dataset.index).then(function(r){return r.json();}).then(function(j){
    var v=s.toLowerCase(), m=j.filter(function(r){return r[0].toLowerCase().indexOf(v)>=0;});
    box.innerHTML = m.length
      ? '<ul class="qlist">'+m.slice(0,200).map(function(r){
          return '<li><a href="'+r[1]+'">'+r[0]+"</a></li>";}).join("")+"</ul>"
      : '<p class="nores">'+box.dataset.none+"</p>";
  }).catch(function(){box.innerHTML='<p class="nores">'+box.dataset.none+"</p>";});
})();
