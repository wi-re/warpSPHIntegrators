// Playback interpolates display samples only; it is not solver dense output.
export function revealSeries(data,time){
 const visible=[];
 for(let i=0;i<data.length;i++){
  const point=data[i];
  if(point[0]<=time){visible.push(point);continue;}
  const prev=data[i-1];
  if(prev&&prev[0]<time&&point[0]>prev[0]&&Number.isFinite(prev[1])&&Number.isFinite(point[1])){
   const alpha=(time-prev[0])/(point[0]-prev[0]);
   visible.push([time,prev[1]+alpha*(point[1]-prev[1])]);
  }
  break;
 }
 return visible;
}
export function pathAppearance(method,highlight){
 const selected=!!highlight&&method===highlight;
 return {alpha:!highlight||selected?1:(method ? .22 : .4),width:selected?3:1.65,selected};
}
