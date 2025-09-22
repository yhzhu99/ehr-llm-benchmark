// generate-manifest.js
const fs = require('fs');
const path = require('path');

const casesDir = path.join(__dirname, 'cases');
const manifest = {};

const groups = ['structured_ehr', 'unstructured_note'];
const tasks = ['mimic-iv-mortality', 'mimic-iv-readmission'];

console.log('Scanning cases directory...');

groups.forEach(group => {
  tasks.forEach(task => {
    const comboId = `${group}::${task}`;
    const dirPath = path.join(casesDir, group, task);

    if (fs.existsSync(dirPath)) {
      const files = fs.readdirSync(dirPath)
        .filter(file => file.toLowerCase().endsWith('.json'))
        .map(file => `cases/${group}/${task}/${file}`); // 生成相对路径

      manifest[comboId] = files;
      console.log(`- Found ${files.length} files for ${comboId}`);
    } else {
      manifest[comboId] = [];
      console.log(`- Directory not found for ${comboId}, skipping.`);
    }
  });
});

const manifestPath = path.join(__dirname, 'file-manifest.json');
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

console.log(`\n✅ Manifest created at ${manifestPath}`);