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
