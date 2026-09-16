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

/* ОДНО ПОЛЕ: ПОИСК ПО ЗАГОЛОВКАМ + ПОМОЩНИК LUKY (17.09). Набор → подсказки из
   /<язык>/search.json (тянется один раз по первому вводу; клик по подсказке — переход).
   Enter/кнопка → POST /api/assistant/ask своего домена (nginx → приложение Luky),
   контракт Luky: {message, history:[{role,text}], country, source}; ответ
   {answer|degraded|empty}. Ответ — плашкой под полем, история на странице; первая
   пара реплик — контекст «какую страницу читает человек». Ни ключей, ни бэкенда. */
(function(){
  var q=document.getElementById("gq"), box=document.getElementById("ask"); if(!q||!box) return;
  var sg=document.getElementById("gsugg"), log=document.getElementById("askLog"), btn=document.getElementById("askBtn");
  var IDX=null, loading=false, busy=false;
  var hist=box.dataset.page?[{role:"user",text:"Я читаю страницу: «"+box.dataset.page+"»"},{role:"model",text:"Понял."}]:[];
  function show(){
    var v=q.value.trim().toLowerCase();
    if(!v||!IDX){sg.style.display="none";return;}
    /* Страны — вперёд: хаб страны узнаём по адресу /<язык>/<страна>/ — ровно два сегмента. */
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
    sg.innerHTML = m.map(function(r){return '<a href="'+r[1]+'">'+r[0]+"</a>";}).join("");
    sg.style.display=m.length?"block":"none";
  }
  function load(){
    if(IDX||loading) return; loading=true;
    fetch(q.dataset.index).then(function(r){return r.json();}).then(function(j){IDX=j;show();})
      .catch(function(){loading=false;});
  }
  function line(cls,text){var d=document.createElement("div");d.className="ask-"+cls;d.textContent=text;log.appendChild(d);return d;}
  function ask(){
    if(busy) return;
    var text=q.value.trim(); if(!text) return;
    q.value=""; sg.style.display="none"; busy=true; box.classList.add("busy");
    line("u",text);
    var w=line("w",box.dataset.wait);
    var body={message:text,history:hist.slice(),country:box.dataset.country||undefined,source:"site"};
    fetch(box.dataset.api,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
      .then(function(r){return r.json();})
      .then(function(d){
        w.remove();
        var a=(d&&!d.degraded&&!d.empty&&d.answer)?d.answer:null;
        line("m",a||box.dataset.err);
        hist.push({role:"user",text:text}); if(a) hist.push({role:"model",text:a});
        if(hist.length>22) hist.splice(hist.length-22,2);
      })
      .catch(function(){w.remove();line("m",box.dataset.err);})
      .then(function(){busy=false;box.classList.remove("busy");q.focus();});
  }
  q.addEventListener("focus",load);
  q.addEventListener("input",function(){load();show();});
  q.addEventListener("keydown",function(e){if(e.key==="Enter"){e.preventDefault();ask();}});
  if(btn) btn.addEventListener("click",ask);
  document.addEventListener("click",function(e){
    if(!e.target.closest("#ask")&&e.target.id!=="gq") sg.style.display="none";
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

