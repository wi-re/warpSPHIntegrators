import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),dist=path.join(root,'dist');
async function files(dir){const found=[];for(const entry of await fs.readdir(dir,{withFileTypes:true})){const name=path.join(dir,entry.name);found.push(...entry.isDirectory()?await files(name):[name]);}return found.sort();}
// Keep imports, worker code and coefficients in one immutable version directory.
// GitHub Pages otherwise permits a ten-minute mixture of cached module versions.
const hash=createHash('sha256');for(const file of [path.join(root,'index.html'),path.join(root,'style.css'),path.join(root,'icon.svg'),...await files(path.join(root,'src'))]){hash.update(path.relative(root,file));hash.update('\0');hash.update(await fs.readFile(file));}
const assetPath='assets/'+hash.digest('hex').slice(0,16),assets=path.join(dist,assetPath);
await fs.rm(dist,{recursive:true,force:true});await fs.mkdir(assets,{recursive:true});
for(const file of ['style.css','icon.svg'])await fs.copyFile(path.join(root,file),path.join(assets,file));
await fs.cp(path.join(root,'src'),path.join(assets,'src'),{recursive:true});
const html=(await fs.readFile(path.join(root,'index.html'),'utf8')).replace('href="./style.css"',`href="./${assetPath}/style.css"`).replace('href="./icon.svg"',`href="./${assetPath}/icon.svg"`).replace('src="./src/app.js"',`src="./${assetPath}/src/app.js"`);
await fs.writeFile(path.join(dist,'index.html'),html);await fs.copyFile(path.join(root,'LICENSE.txt'),path.join(dist,'LICENSE.txt'));
if(process.argv.includes('--docs')){const target=path.join(root,'../docs/static/laboratory');await fs.rm(target,{recursive:true,force:true});await fs.mkdir(target,{recursive:true});await fs.cp(dist,target,{recursive:true});}
console.log('Static site built in web/dist ('+assetPath+')');
