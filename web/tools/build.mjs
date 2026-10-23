import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),dist=path.join(root,'dist');
await fs.mkdir(dist,{recursive:true});
for(const file of ['index.html','style.css','icon.svg'])await fs.copyFile(path.join(root,file),path.join(dist,file));
await fs.cp(path.join(root,'src'),path.join(dist,'src'),{recursive:true});
await fs.copyFile(path.join(root,'LICENSE.txt'),path.join(dist,'LICENSE.txt'));
if(process.argv.includes('--docs')){const target=path.join(root,'../docs/static/laboratory');await fs.mkdir(target,{recursive:true});await fs.cp(dist,target,{recursive:true});}
console.log('Static site built in web/dist');
