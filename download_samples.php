<?php

$conn = new mysqli("91.187.214.100:8306", "root", "marzia69", "corsmagquattro");

$ania = $conn->query("SELECT id FROM corsmagquattro.ricambi WHERE visible = 1 AND id > 0 LIMIT 25")->fetch_all(MYSQLI_ASSOC);

$csv = fopen(__DIR__ . '/metadata.csv', 'w');
fputcsv($csv, ['image_path', 'part_id', 'category_id']);

foreach ($ania as $arow) {
    $sql = "SELECT 
    REPLACE(REPLACE(percorso,
                    '--192.168.0.148-imgcartellini-',
                    ''),
                '-',
                '/') image_path,
    b.idmag part_id,
    c.ania category_id
FROM
    corsmagquattro.foto_cartellini a
        JOIN
    cors_optimized.tag b ON a.cartellino = b.cartellino
        JOIN
    cors_optimized.component c ON b.idmag = c.idmag AND b.idver = c.idver
        JOIN
    corsmagquattro.ricambi d ON c.ania = d.id 
WHERE d.id = $arow[id]
    LIMIT 100";

    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        $total = $result->num_rows;
        $current = 0;
        while ($row = $result->fetch_assoc()) {
            if (!file_exists(__DIR__ . '/samples/' . $row['image_path'])) {
                $image = @file_get_contents('http://91.187.214.100:8380/images/foto/' . $row['image_path']);
                if ($image === false) {
                    echo "Failed to download image: " . $row['image_path'] . "\r";
                    continue; 
                }
                $filePath = __DIR__ . '/samples/' . $row['image_path'];
                $dirPath = dirname($filePath);
                if (!is_dir($dirPath)) {
                    mkdir($dirPath, 0777, true);
                }
                file_put_contents($filePath, $image);
            }
            fputcsv($csv, [$row['image_path'], $row['part_id'], $row['category_id']]);
            $current++;
            echo "Processed $current of $total images.\r";
        }
        echo "Ania {$arow['id']} scritto.\n";
    } else {
        echo "No results found.\n";
    }
}

fclose($csv);